# coding=utf-8
# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MossAudioTokenizer model configuration."""

from typing import Any

try:
    from transformers.configuration_utils import PreTrainedConfig
except ImportError:
    from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MossAudioTokenizerConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MossAudioTokenizerModel`]. It is used to instantiate a
    MossAudioTokenizer model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [OpenMOSS-Team/MOSS-Audio-Tokenizer-v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        sampling_rate (`int`, *optional*, defaults to 48000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        downsample_rate (`int`, *optional*, defaults to 3840):
            Total downsampling rate from waveform to tokens.
        causal_transformer_context_duration (`float`, *optional*, defaults to 10.0):
            Legacy global fallback context duration in seconds for causal transformer. If an individual transformer
            entry in `encoder_kwargs` or `decoder_kwargs` provides `context_duration`, that per-module value takes
            precedence.
        encoder_kwargs (`list[dict]`, *optional*):
            List of encoder module configurations. Each dict specifies a module type and its parameters.
        decoder_kwargs (`list[dict]`, *optional*):
            List of decoder module configurations in execution order.
        number_channels (`int`, *optional*, defaults to 2):
            Number of audio channels exposed by the public waveform interface.
        enable_channel_interleave (`bool`, *optional*, defaults to `True`):
            Whether to flatten multi-channel waveforms into a single internal stream before codec inference.
        attention_implementation (`str`, *optional*, defaults to `"sdpa"`):
            Attention implementation to prefer for transformer layers. Supported values are `"sdpa"` and
            `"flash_attention_2"`.
        compute_dtype (`str`, *optional*, defaults to `"fp32"`):
            Inference compute dtype for non-quantizer modules. Supported values are `"fp32"`, `"bf16"`.
        codec_weight_dtype (`str`, *optional*, defaults to `"fp32"`):
            Parameter dtype for encoder and decoder modules. The quantizer remains fp32 because it explicitly disables
            autocast and performs numerically sensitive codebook operations in fp32.
        quantizer_type (`str`, *optional*, defaults to `"rlfq"`):
            Quantizer type. Options include `"rvq"`, `"spec_rvq"`, `"rlfq"`, `"random_prefix_rlfq"`.
        quantizer_kwargs (`dict`, *optional*):
            Configuration for the quantizer including `input_dim`, `rvq_dim`, `output_dim`, `num_quantizers`,
            `codebook_size`, and `codebook_dim`.

    Example:

    ```python
    >>> from transformers import MossAudioTokenizerModel, MossAudioTokenizerConfig

    >>> # Initializing a MossAudioTokenizer style configuration
    >>> configuration = MossAudioTokenizerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MossAudioTokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "moss-audio-tokenizer"

    # Backward-compatible alias used by some checkpoints.
    attribute_map = {"sample_rate": "sampling_rate"}

    sampling_rate: int
    downsample_rate: int
    causal_transformer_context_duration: float
    encoder_kwargs: list[dict[str, Any]]
    decoder_kwargs: list[dict[str, Any]]
    number_channels: int
    enable_channel_interleave: bool
    attention_implementation: str
    compute_dtype: str
    codec_weight_dtype: str
    quantizer_type: str
    quantizer_kwargs: dict[str, Any]

    def __init__(
        self,
        version: str | None = None,
        sampling_rate: int = 48000,
        downsample_rate: int = 3840,
        causal_transformer_context_duration: float = 10.0,
        encoder_kwargs: list[dict[str, Any]] | None = None,
        decoder_kwargs: list[dict[str, Any]] | None = None,
        number_channels: int = 2,
        enable_channel_interleave: bool = True,
        attention_implementation: str = "sdpa",
        compute_dtype: str = "fp32",
        codec_weight_dtype: str = "fp32",
        quantizer_type: str = "rlfq",
        quantizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        # Some checkpoints might include an incorrect/legacy `model_type` (e.g. "speech_tokenizer").
        # We drop it to avoid overriding the class-level `model_type`.
        kwargs.pop("model_type", None)
        if "channels_numbers" in kwargs:
            number_channels = kwargs.pop("channels_numbers")
        if "enable_channel_interleave" in kwargs:
            enable_channel_interleave = kwargs.pop("enable_channel_interleave")
        if "attention_backend" in kwargs and attention_implementation == "sdpa":
            attention_implementation = kwargs.pop("attention_backend")
        if "codec_compute_dtype" in kwargs and compute_dtype == "fp32":
            compute_dtype = kwargs.pop("codec_compute_dtype")
        if "codec_load_dtype" in kwargs and codec_weight_dtype == "fp32":
            codec_weight_dtype = kwargs.pop("codec_load_dtype")
        reversed_decoder_kwargs = kwargs.pop("reversed_decoder_kwargs", None)

        # `version` is accepted for compatibility but not used in modeling.
        self.version = version
        self.sampling_rate = sampling_rate
        self.downsample_rate = downsample_rate
        self.causal_transformer_context_duration = causal_transformer_context_duration
        self.number_channels = number_channels
        self.enable_channel_interleave = enable_channel_interleave
        self.attention_implementation = attention_implementation
        self.compute_dtype = compute_dtype
        self.codec_weight_dtype = codec_weight_dtype
        # Default encoder configuration
        if encoder_kwargs is None:
            encoder_kwargs = [
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 240,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 240,
                    "output_dimension": 384,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 1.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 384,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 2.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 384,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 4.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 384,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 8.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 640,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 10.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 1280,
                    "output_dimension": 768,
                    "d_model": 1280,
                    "num_heads": 20,
                    "num_layers": 32,
                    "dim_feedforward": 5120,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 10.0,
                },
            ]
        else:
            encoder_kwargs = [dict(module_kwargs) for module_kwargs in encoder_kwargs]
        for module_kwargs in encoder_kwargs:
            if module_kwargs.get("module_type") == "Transformer":
                module_kwargs.setdefault("context_duration", causal_transformer_context_duration)
        self.encoder_kwargs = encoder_kwargs

        # Default decoder configuration (execution order)
        if decoder_kwargs is None and reversed_decoder_kwargs is not None:
            reversed_decoder_kwargs = [dict(module_kwargs) for module_kwargs in reversed_decoder_kwargs]
            decoder_kwargs = []
            for module_kwargs in reversed_decoder_kwargs[::-1]:
                if module_kwargs.get("module_type") != "Transformer":
                    decoder_kwargs.append(module_kwargs)
                    continue
                module_kwargs = dict(module_kwargs)
                module_kwargs["input_dimension"], module_kwargs["output_dimension"] = (
                    module_kwargs["output_dimension"],
                    module_kwargs["input_dimension"],
                )
                decoder_kwargs.append(module_kwargs)

        if decoder_kwargs is None:
            decoder_kwargs = [
                {
                    "module_type": "Transformer",
                    "input_dimension": 768,
                    "output_dimension": 1280,
                    "d_model": 1280,
                    "num_heads": 20,
                    "num_layers": 32,
                    "dim_feedforward": 5120,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 10.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 640,
                    "output_dimension": 768,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 10.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 384,
                    "output_dimension": 768,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 8.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 384,
                    "output_dimension": 768,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 4.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 384,
                    "output_dimension": 768,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 2.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 2,
                },
                {
                    "module_type": "Transformer",
                    "input_dimension": 384,
                    "output_dimension": 240,
                    "d_model": 768,
                    "num_heads": 12,
                    "num_layers": 12,
                    "dim_feedforward": 3072,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                    "context_duration": 1.0,
                },
                {
                    "module_type": "PatchedPretransform",
                    "patch_size": 240,
                },
            ]
        else:
            decoder_kwargs = [dict(module_kwargs) for module_kwargs in decoder_kwargs]
        for module_kwargs in decoder_kwargs:
            if module_kwargs.get("module_type") == "Transformer":
                module_kwargs.setdefault("context_duration", causal_transformer_context_duration)
        self.decoder_kwargs = decoder_kwargs

        # Default quantizer configuration
        if quantizer_kwargs is None:
            quantizer_kwargs = {
                "input_dim": 768,
                "rvq_dim": 512,
                "output_dim": 768,
                "num_quantizers": 32,
                "codebook_size": 1024,
                "codebook_dim": 8,
                "quantizer_type": "rlfq",
            }

        # Handle quantizer_type from kwargs or config
        kw_qtype = quantizer_kwargs.get("quantizer_type", None)
        if kw_qtype is not None:
            self.quantizer_type = kw_qtype
        else:
            self.quantizer_type = quantizer_type
            quantizer_kwargs["quantizer_type"] = quantizer_type

        self.quantizer_kwargs = quantizer_kwargs

        super().__init__(**kwargs)

    @property
    def num_quantizers(self) -> int:
        """Return the number of quantizers from quantizer_kwargs."""
        return self.quantizer_kwargs.get("num_quantizers", 32)

    @property
    def codebook_size(self) -> int:
        """Return the codebook size from quantizer_kwargs."""
        return self.quantizer_kwargs.get("codebook_size", 4096)

    @property
    def frame_rate(self) -> float:
        """Return the frame rate (tokens per second)."""
        return self.sampling_rate / self.downsample_rate


__all__ = ["MossAudioTokenizerConfig"]
