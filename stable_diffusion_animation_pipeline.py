# -------------------------------------------------------------------------------
# This code is copied from 
# https://github.com/andreasjansson/cog-stable-diffusion/blob/animate/animate.py
# -------------------------------------------------------------------------------
from typing import List, Optional, Union, Tuple
import numpy as np

from PIL import Image
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import torch

class StableDiffusionAnimationPipeline(DiffusionPipeline):
    """
    From https://github.com/huggingface/diffusers/pull/241
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        # scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        width: int,
        height: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
    ) -> Image:
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        text_embeddings = self.embed_text(
            prompt, do_classifier_free_guidance, batch_size
        )

        t_start = offset

        ### denoise!

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        safety_cheker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="pt"
        ).to(self.device)
        image, has_nsfw_concept = self.safety_checker(
            images=image, clip_input=safety_cheker_input.pixel_values
        )

        image = self.numpy_to_pil(image)

        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def denoise(self, latents, text_embeddings, t_start, t_end, guidance_scale):
        do_classifier_free_guidance = guidance_scale > 1.0

        for i, t in enumerate(self.scheduler.timesteps[t_start:t_end]):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
        return latents

    def embed_text(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool,
        batch_size: int,
    ) -> torch.FloatTensor:
        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def latents_to_image(self, latents):
        latents = 1 / 0.18215 * latents
        # --------------------------------------------------------------
        # NOTE: ONLY BUG FIX. "sample" was missing in the original code
        # --------------------------------------------------------------
        image = self.vae.decode(latents)["sample"]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image

    def safety_check(self, image):
        safety_cheker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="pt"
        ).to(self.device)
        _, has_nsfw_concept = self.safety_checker(
            images=image, clip_input=safety_cheker_input.pixel_values
        )
        if has_nsfw_concept[0]:
            raise Exception("NSFW content detected, please try a different prompt and/or seed")