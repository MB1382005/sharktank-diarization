"""SharkTank Arabic-English Speaker Diarizer (v1.1).

Built on frozen nvidia/Nemotron-3-Diarization + fine-tuned TemporalResidualAdapter
+ Acoustic-Energy & Collision-Free Math Decision Layer.

Benchmark on Unseen ArzEn Test Split (13 Interviews, 4+ Hours):
  - Raw Nemotron-3 Baseline: 21.84% DER (0.0s collar) | 19.72% DER (0.25s collar) | 20.72% JER
  - Ours (Exp 6):             8.29% DER (0.0s collar) |  7.01% DER (0.25s collar) | 11.21% JER
  - Speaker Count Accuracy:  100% (13/13 recordings exact match)
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, resample_poly, sosfiltfilt
from transformers import AutoModelForAudioFrameClassification, AutoProcessor


class TemporalResidualAdapter(nn.Module):
    """Multi-scale dilated 1D convolutional residual adapter on 10ms speaker logits."""

    def __init__(self, num_speakers: int = 8, hidden: int = 24):
        super().__init__()
        self.conv_in = nn.Conv1d(1, hidden, kernel_size=5, padding=2)
        self.conv_d1 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=10, dilation=5)
        self.conv_d2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=50, dilation=25)
        self.conv_out = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)
        self.cross_spk = nn.Conv1d(num_speakers, num_speakers, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        b_sz, n_frames, n_spks = logits.shape
        x = logits.permute(0, 2, 1).reshape(b_sz * n_spks, 1, n_frames)
        h = F.gelu(self.conv_in(x))
        h = F.gelu(self.conv_d1(h)) + h
        h = F.gelu(self.conv_d2(h)) + h
        delta_temporal = self.conv_out(h).reshape(b_sz, n_spks, n_frames)
        refined = self.cross_spk(logits.permute(0, 2, 1)) + self.alpha * delta_temporal
        return refined.permute(0, 2, 1)


class SharkTankDiarizer:
    """End-to-end calibrated Arabic-English speaker diarizer."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (
            torch.bfloat16
            if (self.device == "cuda" and torch.cuda.is_bf16_supported())
            else (torch.float16 if self.device == "cuda" else torch.float32)
        )
        base_id = "nvidia/Nemotron-3-Diarization"
        self.processor = AutoProcessor.from_pretrained(base_id)
        self.backbone = AutoModelForAudioFrameClassification.from_pretrained(
            base_id, torch_dtype=self.dtype
        ).to(self.device)
        self.backbone.eval()
        self.sr = self.processor.feature_extractor.sampling_rate

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.adapter = TemporalResidualAdapter().to(self.device)
        self.adapter.load_state_dict(ckpt["adapter_state_dict"])
        self.adapter.eval()
        self.lam_star = float(ckpt.get("lam_star", 0.4))
        self.cfg_star = ckpt.get("cfg_star", {})

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "Mbahaa08/SharkTank-ArabicEnglish-Diarizer",
        token: str | None = None,
    ) -> "SharkTankDiarizer":
        local_ckpt = Path(__file__).resolve().parent / "nemotron3_arzen_adapter.pt"
        if local_ckpt.exists():
            return cls(local_ckpt)
        ckpt_path = hf_hub_download(
            repo_id=repo_id, filename="nemotron3_arzen_adapter.pt", token=token
        )
        return cls(ckpt_path)

    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        x = audio - np.mean(audio)
        sos = butter(4, [80.0, 7600.0], btype="bandpass", fs=self.sr, output="sos")
        x = sosfiltfilt(sos, x).astype(np.float32)
        win = int(0.05 * self.sr)
        env = np.sqrt(uniform_filter1d(x ** 2, size=win) + 1e-10)
        noise_floor, target_rms = np.percentile(env, 20), np.percentile(env, 85)
        speech_mask = np.clip((env - 2.0 * noise_floor) / (noise_floor + 1e-8), 0.0, 1.0)
        desired_gain = np.clip(target_rms / (env + 1e-6), 1.0, 3.0)
        gain = 1.0 + speech_mask * (desired_gain - 1.0)
        x_boosted = x * gain
        x_norm = x_boosted / (np.max(np.abs(x_boosted)) + 1e-8)
        return (0.95 * np.tanh(1.4 * x_norm) / np.tanh(1.4)).astype(np.float32)

    def diarize(
        self, audio_path: str | Path, output_rttm: str | Path | None = None
    ) -> list[dict]:
        audio, orig_sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != self.sr:
            g = gcd(orig_sr, self.sr)
            audio = resample_poly(audio, self.sr // g, orig_sr // g).astype(np.float32)

        raw_audio = audio.copy()
        pre_audio = self.preprocess_audio(audio)
        self.processor.set_streaming_mode("low_latency")

        def chunk_gen():
            yield self.processor(
                pre_audio[: self.processor.num_samples_first_audio_chunk],
                sampling_rate=self.sr,
                is_streaming=True,
                is_first_audio_chunk=True,
            )
            mel_idx = self.processor.num_mel_frames_per_step
            start_idx = self.processor.audio_chunk_start(mel_idx)
            while (
                end_idx := start_idx + self.processor.num_samples_per_audio_chunk
            ) <= pre_audio.shape[0]:
                yield self.processor(
                    pre_audio[start_idx:end_idx],
                    sampling_rate=self.sr,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                )
                mel_idx += self.processor.num_mel_frames_per_step
                start_idx = self.processor.audio_chunk_start(mel_idx)
            if start_idx < pre_audio.shape[0]:
                yield self.processor(
                    pre_audio[start_idx:],
                    sampling_rate=self.sr,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                    is_last_audio_chunk=True,
                )

        speaker_cache, all_logits = None, []
        with torch.inference_mode():
            for inputs in chunk_gen():
                inputs = inputs.to(self.device, dtype=self.dtype)
                outputs = self.backbone(**inputs, speaker_cache=speaker_cache)
                all_logits.append(outputs.logits.cpu())
                speaker_cache = outputs.speaker_cache

            z_raw = torch.cat(all_logits, dim=1).float()
            z_ft = self.adapter(z_raw.to(self.device)).cpu()
            z_blend = (1.0 - self.lam_star) * z_raw + self.lam_star * z_ft

        probs = torch.sigmoid(z_blend[0]).numpy()
        n_frames, n_spks = probs.shape
        dt = 0.01
        probs = gaussian_filter1d(probs, sigma=4.0, axis=0)

        hop, win = int(0.01 * self.sr), int(0.03 * self.sr)
        rms = np.sqrt(uniform_filter1d(raw_audio ** 2, size=win) + 1e-10)[::hop]
        rms = np.pad(rms, (0, max(0, n_frames - len(rms))), mode="edge")[:n_frames]
        p20, p85 = np.percentile(rms, 20), np.percentile(rms, 85)
        energy = np.clip((rms - p20) / (p85 - p20 + 1e-8), 0.0, 1.0)
        top_spk = np.argmax(probs, axis=1)
        for t in range(n_frames):
            d = top_spk[t]
            if probs[t, d] > 0.12:
                probs[t, d] = min(1.0, probs[t, d] + 0.08 * energy[t])

        raw_by_spk, total_dur = {}, {}
        for spk in range(n_spks):
            p = probs[:, spk]
            active = np.zeros(n_frames, dtype=bool)
            is_on = False
            for t in range(n_frames):
                if not is_on and p[t] >= 0.44:
                    is_on = True
                elif is_on and p[t] < 0.24:
                    is_on = False
                active[t] = is_on
            if not np.any(active):
                continue
            padded = np.concatenate([[False], active, [False]])
            starts = np.where(padded[1:] & ~padded[:-1])[0]
            ends = np.where(~padded[1:] & padded[:-1])[0]
            segs = []
            for s_f, e_f in zip(starts, ends):
                if (e_f - s_f) < 5:
                    continue
                new_s, new_e = s_f, e_f
                for f in range(s_f - 1, max(0, s_f - 7) - 1, -1):
                    if np.max(np.delete(probs[f], spk)) > 0.55:
                        break
                    new_s = f
                for f in range(e_f, min(n_frames, e_f + 13)):
                    if np.max(np.delete(probs[f], spk)) > 0.55:
                        break
                    new_e = f + 1
                segs.append([new_s * dt, new_e * dt])
            if not segs:
                continue
            merged = [segs[0]]
            for s, e in segs[1:]:
                if (s - merged[-1][1]) <= 1.25:
                    g1, g2 = int(merged[-1][1] / dt), int(s / dt)
                    other_ratio = (
                        float(
                            np.mean(
                                np.max(np.delete(probs[g1:g2], spk, axis=1), axis=1)
                                > 0.60
                            )
                        )
                        if g2 > g1
                        else 0.0
                    )
                    if other_ratio < 0.50:
                        merged[-1][1] = max(merged[-1][1], e)
                    else:
                        merged.append([s, e])
                else:
                    merged.append([s, e])
            merged = [[s, e] for s, e in merged if (e - s) >= 0.15]
            if merged:
                raw_by_spk[spk] = merged
                total_dur[spk] = sum(e - s for s, e in merged)

        major = [k for k, d in total_dur.items() if d >= 15.0] or list(total_dur.keys())
        final_segments = []
        for spk, segs in raw_by_spk.items():
            for s, e in segs:
                target_spk = (
                    spk
                    if spk in major
                    else max(
                        major,
                        key=lambda m: float(
                            probs[max(0, int(s / dt)) : min(n_frames, int(e / dt)), m].mean()
                        ),
                    )
                )
                final_segments.append(
                    {"Start": round(s, 3), "End": round(e, 3), "Speaker": int(target_spk)}
                )

        final_segments.sort(key=lambda x: (x["Start"], x["End"]))
        if output_rttm:
            uri = Path(audio_path).stem
            with open(output_rttm, "w", encoding="utf-8") as f:
                for seg in final_segments:
                    dur = seg["End"] - seg["Start"]
                    if dur > 0:
                        f.write(
                            f"SPEAKER {uri} 1 {seg['Start']:.3f} {dur:.3f} <NA> <NA> speaker_{seg['Speaker']} <NA> <NA>\n"
                        )
        return final_segments
