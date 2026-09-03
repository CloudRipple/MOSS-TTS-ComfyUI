# MOSS-TTS-V15-ComfyUI patch notes (vs upstream diffsynth/models/dac_vae.py):
#   * descript-audiotools dependency removed: no audiotools.AudioSignal, no
#     audiotools.ml.BaseModel. DAC is a plain nn.Module.
#   * .dac container I/O removed: DACFile, CodecMixin.compress/decompress (they
#     exist for the audio-codec use case, not for latent-space VAE inference).
#   * Discrete-codec machinery removed: VectorQuantize / ResidualVectorQuantize,
#     DiracDistribution and the forward()/KL paths. MOSS-SoundEffect's
#     vae_128d_48k checkpoint is continuous (Gaussian-posterior latent), so only
#     the continuous branch is kept; constructing with continuous=False raises.
#   * DAC.load is our own small loader for the torch.load() package format
#     ({'state_dict', 'metadata': {'kwargs'}}), replacing BaseModel.load.
import math
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


class CodecMixin:
    # MOSS-TTS-V15-ComfyUI patch: padding property removed (only compress()
    # toggled it); get_delay/get_output_length are kept because DAC.__init__
    # computes self.delay from them.
    def get_delay(self):
        # Any number works here, delay is invariant to input length
        l_out = self.get_output_length(0)
        L = l_out

        layers = []
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(layer)

        for layer in reversed(layers):
            d = layer.dilation[0]
            k = layer.kernel_size[0]
            s = layer.stride[0]

            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1

            L = math.ceil(L)

        l_in = L

        return (l_in - l_out) // 2

    def get_output_length(self, input_length):
        L = input_length
        # Calculate output length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]

                if isinstance(layer, nn.Conv1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d):
                    L = (L - 1) * s + d * (k - 1) + 1

                L = math.floor(L)
        return L


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def mode(self):
        return self.mean


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(nn.Module, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        continuous: bool = False,
    ):
        super().__init__()
        # MOSS-TTS-V15-ComfyUI patch: the discrete RVQ branch is not vendored
        # (inference-only trim); the n_codebooks/codebook_* args are accepted so
        # the packaged checkpoint metadata can be replayed verbatim.
        if not continuous:
            raise NotImplementedError(
                "This vendored DAC only supports continuous=True "
                "(the MOSS-SoundEffect vae_128d_48k checkpoint is continuous)."
            )

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.continuous = continuous

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        # MOSS-TTS-V15-ComfyUI patch: math.prod instead of numpy.prod, so numpy
        # is not a dependency of this module.
        self.hop_length = math.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.quant_conv = torch.nn.Conv1d(latent_dim, 2 * latent_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(latent_dim, latent_dim, 1)

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

    @classmethod
    def load(cls, load_path) -> "DAC":
        """Load a packaged DAC checkpoint.

        MOSS-TTS-V15-ComfyUI patch: replaces audiotools.ml.BaseModel.load so we
        do not depend on descript-audiotools. The upstream package is a
        torch.load()able dict of the form
            {"state_dict": {...}, "metadata": {"kwargs": {<ctor args>}}}
        The stored state_dict uses the legacy weight_norm layout
        (``weight_g``/``weight_v``); torch's parametrization-based weight_norm
        maps those onto ``parametrizations.weight.original0/1`` through its
        load-state-dict pre-hook, so a strict load succeeds.
        """
        load_path = str(load_path)
        if load_path.endswith(".safetensors"):
            raise ValueError(
                f"Cannot build a DAC from a bare safetensors file ({load_path}): "
                f"the constructor args live inside the packaged .pth metadata. "
                f"Point the pipeline at vae_128d_48k.pth."
            )
        package = torch.load(load_path, map_location="cpu", weights_only=True)
        model = cls(**package["metadata"]["kwargs"])
        model.load_state_dict(package["state_dict"], strict=True)
        return model

    @property
    def dtype(self):
        """Get the dtype of the model parameters."""
        # Return the dtype of the first parameter found
        for param in self.parameters():
            return param.dtype
        return torch.float32  # fallback

    @property
    def device(self):
        """Get the device of the model parameters."""
        # Return the device of the first parameter found
        for param in self.parameters():
            return param.device
        return torch.device('cpu')  # fallback

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
    ):
        """Encode audio data and return the Gaussian posterior.

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
        n_quantizers : unused (kept for upstream signature parity)

        Returns
        -------
        (posterior, None, None, 0, 0)
            posterior is a DiagonalGaussianDistribution; call .mode() or
            .sample() to get latents of shape [B x latent_dim x T].
        """
        z = self.encoder(audio_data)  # [B x D x T]
        z = self.quant_conv(z)  # [B x 2D x T]
        z = DiagonalGaussianDistribution(z)
        codes, latents, commitment_loss, codebook_loss = None, None, 0, 0

        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor):
        """Decode latents of shape [B x latent_dim x T] into audio [B x 1 x time]."""
        z = self.post_quant_conv(z)
        audio = self.decoder(z)
        return audio
