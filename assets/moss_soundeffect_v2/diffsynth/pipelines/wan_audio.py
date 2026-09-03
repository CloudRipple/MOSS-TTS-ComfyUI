# MOSS-TTS-V15-ComfyUI patch notes (vs upstream diffsynth/pipelines/wan_audio.py):
#   * diffusers dependency removed (no AutoencoderOobleck): the MOSS-SoundEffect
#     checkpoint is always the DAC VAE, so the oobleck branches are gone and the
#     VAE loads only from the packaged vae_128d_48k.pth.
#   * Video-only / acceleration machinery removed: TeaCache, dit2 + DiT-boundary
#     switching, WanVideoUnit_UnifiedSequenceParallel (xfuser),
#     WanVideoUnit_CfgMerger + cfg_merge, TemporalTiler + sliding-window, VACE,
#     motion_controller, training_loss(), gradient-checkpointing branches in
#     model_fn_wan_video, torch.compiler.cudagraph_mark_step_begin, and the
#     @torch.compile decorator on model_fn_wan_video (ComfyUI manages
#     compilation/device placement itself).
#   * numpy / PIL / einops imports removed with the deleted pieces.
#   * sinusoidal_embedding_1d now comes from ..models.dit_block (upstream
#     imported it from the non-vendored wan_video_dit.py).
import torch, os, json
from typing import Optional, Union
from tqdm import tqdm
from safetensors.torch import load_file

from ..utils import BasePipeline, PipelineUnit, PipelineUnitRunner
from ..models.wan_audio_dit import WanAudioModel
from ..models.dit_block import sinusoidal_embedding_1d
from ..models.qwen3_text_encoder import Qwen3TextEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..models.dac_vae import DAC


# Maps diffusers-style keys in the exported HF DiT checkpoint back to the
# native WanAudioModel keys. Paired with the forward direction in
# moss_soundeffect_v2/hf_export.py.
_HF_DIT_BLOCK_RENAME = {
    "attn1.norm_k.weight": "self_attn.norm_k.weight",
    "attn1.norm_q.weight": "self_attn.norm_q.weight",
    "attn1.to_k.bias": "self_attn.k.bias",
    "attn1.to_k.weight": "self_attn.k.weight",
    "attn1.to_out.0.bias": "self_attn.o.bias",
    "attn1.to_out.0.weight": "self_attn.o.weight",
    "attn1.to_q.bias": "self_attn.q.bias",
    "attn1.to_q.weight": "self_attn.q.weight",
    "attn1.to_v.bias": "self_attn.v.bias",
    "attn1.to_v.weight": "self_attn.v.weight",
    "attn2.norm_k.weight": "cross_attn.norm_k.weight",
    "attn2.norm_q.weight": "cross_attn.norm_q.weight",
    "attn2.to_k.bias": "cross_attn.k.bias",
    "attn2.to_k.weight": "cross_attn.k.weight",
    "attn2.to_out.0.bias": "cross_attn.o.bias",
    "attn2.to_out.0.weight": "cross_attn.o.weight",
    "attn2.to_q.bias": "cross_attn.q.bias",
    "attn2.to_q.weight": "cross_attn.q.weight",
    "attn2.to_v.bias": "cross_attn.v.bias",
    "attn2.to_v.weight": "cross_attn.v.weight",
    "ffn.net.0.proj.bias": "ffn.0.bias",
    "ffn.net.0.proj.weight": "ffn.0.weight",
    "ffn.net.2.bias": "ffn.2.bias",
    "ffn.net.2.weight": "ffn.2.weight",
    "norm2.bias": "norm3.bias",
    "norm2.weight": "norm3.weight",
    "scale_shift_table": "modulation",
}

_HF_DIT_GLOBAL_RENAME = {
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "patch_embedding.weight": "patch_embedding.weight",
}


def _convert_hf_dit_state_dict(state_dict: dict) -> dict:
    out = {}
    for key, param in state_dict.items():
        if key in _HF_DIT_GLOBAL_RENAME:
            out[_HF_DIT_GLOBAL_RENAME[key]] = param
        elif key.startswith("blocks."):
            parts = key.split(".", 2)
            block_idx, suffix = parts[1], parts[2]
            if suffix in _HF_DIT_BLOCK_RENAME:
                out[f"blocks.{block_idx}.{_HF_DIT_BLOCK_RENAME[suffix]}"] = param
            else:
                out[key] = param
        else:
            out[key] = param
    return out


class WanAudioPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None, flow_shift=5.0):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=flow_shift, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder = None
        self.image_encoder = None
        self.dit: WanAudioModel = None
        self.vae: DAC = None
        # MOSS-TTS-V15-ComfyUI patch: no dit2 / motion_controller / vace.
        self.in_iteration_models = ("dit",)
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanAudioUnit_ShapeChecker(),
            WanAudioUnit_NoiseInitializer(),
            WanAudioUnit_InputAudioEmbedder(),
            WanVideoUnit_PromptEmbedder(),
        ]
        self.model_fn = model_fn_wan_video

    # MOSS-TTS-V15-ComfyUI patch: upstream training_loss() is not vendored
    # (inference-only package).

    def check_resize_num_channels_num_samples(self, num_channels, num_samples):
        # Shape check
        if num_samples % self.num_samples_division_factor != 0:
            num_samples = num_samples // self.num_samples_division_factor * self.num_samples_division_factor
        return num_channels, num_samples


    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: Union[str, torch.device] = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "WanAudioPipeline":
        """Load a WanAudioPipeline from a HuggingFace-format directory.

        Expected layout (a diffusers-style HF model directory):
            model_dir/
                model_index.json
                scheduler/scheduler_config.json
                transformer/config.json
                transformer/diffusion_pytorch_model.safetensors
                text_encoder/...        (Qwen3)
                tokenizer/...
                vae/vae_128d_48k.pth
        """
        with open(os.path.join(model_dir, "model_index.json")) as f:
            index = json.load(f)
        print(f"Loading from: {model_dir}")
        print(f"  Pipeline: {index['_class_name']}, dit_variant: {index.get('dit_variant')}")

        with open(os.path.join(model_dir, "scheduler", "scheduler_config.json")) as f:
            sched_cfg = json.load(f)
        with open(os.path.join(model_dir, "transformer", "config.json")) as f:
            dit_cfg = json.load(f)

        te_path = os.path.join(model_dir, "text_encoder")
        print(f"  Loading text_encoder from {te_path} ...")
        text_encoder = Qwen3TextEncoder(te_path, torch_dtype=torch_dtype)
        text_encoder = text_encoder.to(device)
        print(f"  text_encoder: dim={text_encoder.dim}")

        tok_path = os.path.join(model_dir, "tokenizer")
        print(f"  Loading tokenizer from {tok_path} ...")
        prompter = WanPrompter(tokenizer_path=tok_path)
        prompter.fetch_models(text_encoder)

        vae_dir = os.path.join(model_dir, "vae")
        vae_pth = os.path.join(vae_dir, "vae_128d_48k.pth")
        # MOSS-TTS-V15-ComfyUI patch: upstream also looked for a diffusers-style
        # diffusion_pytorch_model.safetensors here; without the .pth package
        # format there are no DAC constructor args available, so only the
        # packaged checkpoint is supported.
        if not os.path.exists(vae_pth):
            raise FileNotFoundError(f"No packaged DAC VAE (vae_128d_48k.pth) found in {vae_dir}")
        print(f"  Loading DAC VAE from {vae_pth} ...")
        vae = DAC.load(vae_pth)

        dit_weights_path = os.path.join(model_dir, "transformer", "diffusion_pytorch_model.safetensors")
        print(f"  Loading DiT from {dit_weights_path} ...")
        diffusers_sd = load_file(dit_weights_path)
        custom_sd = _convert_hf_dit_state_dict(diffusers_sd)

        dit = WanAudioModel(
            in_dim=dit_cfg["in_dim"],
            out_dim=dit_cfg["out_dim"],
            text_dim=dit_cfg["text_dim"],
            freq_dim=dit_cfg["freq_dim"],
            eps=dit_cfg["eps"],
            patch_size=tuple(dit_cfg["patch_size"]),
            has_image_input=dit_cfg["has_image_input"],
            dim=dit_cfg["dim"],
            ffn_dim=dit_cfg["ffn_dim"],
            num_heads=dit_cfg["num_heads"],
            num_layers=dit_cfg["num_layers"],
            vae_type=dit_cfg.get("vae_type", "dac"),
        )
        load_result = dit.load_state_dict(custom_sd)
        print(
            f"  DiT loaded: missing={len(load_result.missing_keys)}, "
            f"unexpected={len(load_result.unexpected_keys)}"
        )

        pipe = cls(
            device=device,
            torch_dtype=torch_dtype,
            flow_shift=sched_cfg.get("shift", 5.0),
        )
        pipe.text_encoder = text_encoder
        pipe.prompter = prompter
        pipe.vae = vae
        pipe.dit = dit
        pipe.audio_latent_dim = dit_cfg["in_dim"]
        pipe.num_samples_division_factor = vae.hop_length
        pipe.dit_variant = index.get("dit_variant")
        pipe.to(device)
        print(f"  Pipeline assembled on {device}")
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: Union[str, list[str]],
        negative_prompt: Optional[Union[str, list[str]]] = "",
        denoising_strength: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        num_samples=44100*10,
        num_channels=2,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
        }
        # Infer batch size; prompt may be a list[str].
        computed_batch_size = len(prompt) if isinstance(prompt, (list, tuple)) else 1
        # For batched input, broadcast a single negative_prompt to the same length.
        if computed_batch_size > 1 and not isinstance(negative_prompt, (list, tuple)):
            inputs_nega["negative_prompt"] = [negative_prompt] * computed_batch_size
        inputs_shared = {
            "num_samples": num_samples,
            "num_channels": num_channels,
            "denoising_strength": denoising_strength,
            "seed": seed, "rand_device": rand_device,
            "cfg_scale": cfg_scale,
            "sigma_shift": sigma_shift,
            "batch_size": computed_batch_size,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Timestep
            timestep = timestep.unsqueeze(0).to(device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_posi = noise_pred_posi.clone()
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred_posi = noise_pred_posi.float()
                noise_pred_nega = noise_pred_nega.float()
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        # Decode (in batches of at most max_decode_bs latents)
        self.load_models_to_device(['vae'])
        latents = inputs_shared["latents"]
        max_decode_bs = 8
        audio_chunks = []
        for start in range(0, latents.size(0), max_decode_bs):
            end = min(start + max_decode_bs, latents.size(0))
            with torch.autocast("cuda", dtype=torch.float32):
                # MOSS-TTS-V15-ComfyUI patch: DAC-only decode branch.
                audio_chunk = self.vae.decode(latents[start:end])
            audio_chunks.append(audio_chunk)
        audio = torch.cat(audio_chunks, dim=0)
        self.load_models_to_device([])

        return audio



class WanAudioUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("num_channels", "num_samples"))

    def process(self, pipe: WanAudioPipeline, num_channels, num_samples):
        num_channels, num_samples = pipe.check_resize_num_channels_num_samples(num_channels, num_samples)
        return {"num_channels": num_channels, "num_samples": num_samples}



class WanAudioUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("input_audio", "num_samples", "seed", "rand_device", "batch_size"))

    def process(self, pipe: WanAudioPipeline, input_audio, num_samples, seed, rand_device, batch_size):
        if input_audio is not None:
            bsz = input_audio.size(0) if input_audio.ndim == 3 else 1
        else:
            bsz = batch_size if batch_size is not None else 1
        shape = (bsz, pipe.audio_latent_dim, num_samples // pipe.num_samples_division_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}



class WanAudioUnit_InputAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_audio", "audio_latent", "noise"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanAudioPipeline, input_audio, audio_latent, noise):
        # Pass-through branch when an audio_latent is provided directly.
        if audio_latent is not None:
            latents = audio_latent
            if latents.ndim == 2:
                latents = latents.unsqueeze(0)
            latents = latents.to(dtype=pipe.torch_dtype, device=pipe.device)
            if pipe.scheduler.training:
                return {"latents": noise, "input_latents": latents}
            else:
                latents = pipe.scheduler.add_noise(latents, noise, timestep=pipe.scheduler.timesteps[0])
                return {"latents": latents}

        if input_audio is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        if input_audio.ndim == 2:
            # add batch dim
            input_audio = input_audio.unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.float32):
            # MOSS-TTS-V15-ComfyUI patch: DAC-only encode branch.
            input_latents = pipe.vae.encode(input_audio)[0].mode()
        input_latents = input_latents.to(device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanAudioPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}


# MOSS-TTS-V15-ComfyUI patch: model_fn_wan_video keeps upstream math but drops
# the @torch.compile decorator (upstream: options={"triton.cudagraphs": True},
# fullgraph=True) plus every branch the audio inference path cannot reach
# (TeaCache, VACE, UnifiedSequenceParallel/xfuser, sliding window/TemporalTiler,
# motion-controller, separated-timestep TI2V path, gradient checkpointing).
def model_fn_wan_video(
    dit: WanAudioModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    control_camera_latents_input = None,
    **kwargs,
):
    # Timestep
    with torch.autocast("cuda", dtype=torch.float32):
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding (never produced by the audio units; kept for parity with
    # the WanAudioModel conditioning fields)
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x, (f, ) = dit.patchify(x, control_camera_latents_input)

    # Reference audio/image latents (unit never emits them; kept for parity)
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    # freqs is now a registered buffer (moves with model.to(device)). Do not
    # write Python attributes here so torch.compile can trace through.
    audio_freqs = dit.freqs
    freqs = torch.cat([
        audio_freqs[0][:f].view(f, -1).expand(f, -1),
        audio_freqs[1][:f].view(f, -1).expand(f, -1),
        audio_freqs[2][:f].view(f, -1).expand(f, -1),
    ], dim=-1).reshape(f, 1, -1)

    # blocks
    for block in dit.blocks:
        x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, ))
    return x
