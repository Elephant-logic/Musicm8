from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
import torchaudio


@dataclass
class CodecInfo:
    name: str
    sample_rate: int
    frame_rate: float
    channels: int
    codebook_size: int
    num_codebooks: int
    stereo_interleaved: bool = False
    bandwidth: float | None = None
    model_type: str | None = None
    n_quantizers: int | None = None
    custom_checkpoint: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class AudioCodec(Protocol):
    info: CodecInfo
    def encode(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor: ...
    def decode(self, codes: torch.Tensor) -> torch.Tensor: ...


def _resample_channels(wav: torch.Tensor, sr: int, target_sr: int, channels: int) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if channels == 1 and wav.shape[0] != 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif channels == 2:
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
    return wav


class Encodec24Codec:
    def __init__(self, device: torch.device, bandwidth: float = 3.0, channels: int = 1):
        try:
            from encodec import EncodecModel
        except ImportError as e:
            raise RuntimeError("Install encodec (`pip install encodec`) to use EnCodec.") from e
        self.device = device
        self.channels = channels
        self.model = EncodecModel.encodec_model_24khz().to(device).eval()
        self.model.set_target_bandwidth(bandwidth)
        q_mono = {1.5: 2, 3.0: 4, 6.0: 8, 12.0: 16, 24.0: 32}[float(bandwidth)]
        self.info = CodecInfo(name="encodec24", sample_rate=int(self.model.sample_rate), frame_rate=75.0, channels=channels, codebook_size=1024, num_codebooks=q_mono * channels, stereo_interleaved=channels == 2, bandwidth=float(bandwidth))

    @torch.no_grad()
    def _encode_mono(self, mono: torch.Tensor) -> torch.Tensor:
        frames = self.model.encode(mono.unsqueeze(0).to(self.device))
        return torch.cat([c for c, _ in frames], dim=-1)[0]

    def encode(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav = _resample_channels(wav, sample_rate, self.info.sample_rate, self.channels)
        if self.channels == 1:
            return self._encode_mono(wav).cpu()
        left = self._encode_mono(wav[0:1]); right = self._encode_mono(wav[1:2])
        return torch.stack([x for pair in zip(left, right) for x in pair], dim=0).cpu()

    @torch.no_grad()
    def _decode_mono(self, codes: torch.Tensor) -> torch.Tensor:
        # Token files are stored as int16 to save Drive space, but EnCodec's
        # codebook lookup expects integer index tensors (int32/int64).
        codes = codes.long()
        return self.model.decode([(codes.unsqueeze(0).to(self.device), None)])[0]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim == 3:
            if codes.shape[0] != 1: raise ValueError("decode currently supports batch size 1")
            codes = codes[0]
        if self.channels == 1: return self._decode_mono(codes).cpu()
        l = self._decode_mono(codes[0::2])[0]; r = self._decode_mono(codes[1::2])[0]
        return torch.stack([l, r], dim=0).cpu()


class HFEncodec32Codec:
    def __init__(self, device: torch.device, channels: int = 1):
        try:
            from transformers import EncodecModel
        except ImportError as e:
            raise RuntimeError("Install transformers to use facebook/encodec_32khz.") from e
        self.device = device; self.channels = channels
        self.model = EncodecModel.from_pretrained("facebook/encodec_32khz").to(device).eval()
        self.info = CodecInfo(name="encodec32", sample_rate=32000, frame_rate=50.0, channels=channels, codebook_size=int(self.model.config.codebook_size), num_codebooks=4 * channels, stereo_interleaved=channels == 2, bandwidth=2.2, model_type="facebook/encodec_32khz")

    @torch.no_grad()
    def _encode_mono(self, mono: torch.Tensor) -> torch.Tensor:
        x = mono.unsqueeze(0).to(self.device); mask = torch.ones_like(x, dtype=torch.bool)
        out = self.model.encode(x, padding_mask=mask, bandwidth=2.2, return_dict=True)
        return out.audio_codes[0, 0]

    def encode(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav = _resample_channels(wav, sample_rate, self.info.sample_rate, self.channels)
        if self.channels == 1: return self._encode_mono(wav).cpu()
        left, right = self._encode_mono(wav[0:1]), self._encode_mono(wav[1:2])
        return torch.stack([x for pair in zip(left, right) for x in pair], dim=0).cpu()

    @torch.no_grad()
    def _decode_mono(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.long()
        out = self.model.decode(codes.unsqueeze(0).unsqueeze(0).to(self.device), [None], return_dict=True)
        return out.audio_values[0]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim == 3:
            if codes.shape[0] != 1: raise ValueError("decode currently supports batch size 1")
            codes = codes[0]
        if self.channels == 1: return self._decode_mono(codes).cpu()
        l = self._decode_mono(codes[0::2])[0]; r = self._decode_mono(codes[1::2])[0]
        return torch.stack([l, r], dim=0).cpu()


class DACCodec:
    def __init__(self, device: torch.device, model_type: str = "44khz", n_quantizers: int | None = None, channels: int = 1):
        try:
            import dac
        except ImportError as e:
            raise RuntimeError("Install descript-audio-codec (`pip install descript-audio-codec`).") from e
        self.device = device; self.channels = channels
        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path).to(device).eval()
        q_mono = int(n_quantizers or self.model.n_codebooks)
        if q_mono > int(self.model.n_codebooks): raise ValueError(f"n_quantizers={q_mono} exceeds DAC capacity {self.model.n_codebooks}")
        self.n_quantizers = q_mono
        self.info = CodecInfo(name="dac", sample_rate=int(self.model.sample_rate), frame_rate=float(self.model.sample_rate / self.model.hop_length), channels=channels, codebook_size=int(self.model.codebook_size), num_codebooks=q_mono * channels, stereo_interleaved=channels == 2, model_type=model_type, n_quantizers=q_mono)

    @torch.no_grad()
    def _encode_mono(self, mono: torch.Tensor) -> torch.Tensor:
        x = self.model.preprocess(mono.unsqueeze(0).to(self.device), self.info.sample_rate)
        _, codes, _, _, _ = self.model.encode(x, n_quantizers=self.n_quantizers)
        return codes[0]

    def encode(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav = _resample_channels(wav, sample_rate, self.info.sample_rate, self.channels)
        if self.channels == 1: return self._encode_mono(wav).cpu()
        left = self._encode_mono(wav[0:1]); right = self._encode_mono(wav[1:2])
        return torch.stack([x for pair in zip(left, right) for x in pair], dim=0).cpu()

    @torch.no_grad()
    def _decode_mono(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.long()
        out = self.model.quantizer.from_codes(codes.unsqueeze(0).to(self.device))
        z = out[0] if isinstance(out, (tuple, list)) else out
        return self.model.decode(z)[0]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim == 3:
            if codes.shape[0] != 1: raise ValueError("decode currently supports batch size 1")
            codes = codes[0]
        if self.channels == 1: return self._decode_mono(codes).cpu()
        l = self._decode_mono(codes[0::2])[0]; r = self._decode_mono(codes[1::2])[0]
        return torch.stack([l, r], dim=0).cpu()


class CustomMusicCodec:
    def __init__(self, device: torch.device, checkpoint: str, channels: int | None = None):
        from music_codec import MusicCodecConfig, MusicSemanticAcousticCodec
        self.device = device
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = MusicCodecConfig.from_dict(ckpt["config"])
        if channels is not None and int(channels) != cfg.channels: raise ValueError(f"custom codec was trained for {cfg.channels} channel(s), requested {channels}")
        self.model = MusicSemanticAcousticCodec(cfg).to(device).eval(); self.model.load_state_dict(ckpt["model"])
        self.cfg = cfg
        self.info = CodecInfo(name="musiccodec", sample_rate=cfg.sample_rate, frame_rate=cfg.frame_rate, channels=cfg.channels, codebook_size=cfg.codebook_size, num_codebooks=1 + cfg.acoustic_codebooks, stereo_interleaved=False, model_type="semantic_acoustic_vq", custom_checkpoint=str(Path(checkpoint).resolve()))

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav = _resample_channels(wav, sample_rate, self.info.sample_rate, self.info.channels)
        return self.model.codes_from_audio(wav.unsqueeze(0).to(self.device))[0].cpu()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim == 2: codes = codes.unsqueeze(0)
        if codes.shape[0] != 1: raise ValueError("decode currently supports batch size 1")
        return self.model.decode_codes(codes.long().to(self.device))[0].cpu()


def create_codec(name: str, device: torch.device, *, channels: int = 1, bandwidth: float = 3.0, model_type: str = "44khz", n_quantizers: int | None = None, custom_checkpoint: str | None = None) -> AudioCodec:
    if name == "encodec24": return Encodec24Codec(device, bandwidth=bandwidth, channels=channels)
    if name == "encodec32": return HFEncodec32Codec(device, channels=channels)
    if name == "dac": return DACCodec(device, model_type=model_type, n_quantizers=n_quantizers, channels=channels)
    if name == "musiccodec":
        if not custom_checkpoint: raise ValueError("musiccodec requires --custom-codec checkpoint")
        return CustomMusicCodec(device, custom_checkpoint, channels=channels)
    raise ValueError(f"Unknown codec {name!r}")


def codec_from_info(info: dict, device: torch.device) -> AudioCodec:
    return create_codec(info["name"], device, channels=int(info.get("channels", 1)), bandwidth=float(info.get("bandwidth") or 3.0), model_type=str(info.get("model_type") or "44khz"), n_quantizers=info.get("n_quantizers"), custom_checkpoint=info.get("custom_checkpoint"))
