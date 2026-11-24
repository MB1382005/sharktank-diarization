"""Generate standard RTTM ground-truth files from ArzEn_SpeechCorpus_1.0.

Each ArzEn interview recording (e.g., C03-R04.WAV) contains Egyptian Arabic +
English code-switched multi-speaker speech (typically 3 speakers: I01, I02, Sxx)
at 16 kHz Mono WAV.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

ARZEN_ROOT = Path(r"D:\Grad\DataSet\ArzEn_SpeechCorpus_1.0\ArzEn_SpeechCorpus_1.0")
OUT_ROOT = Path(r"D:\Grad\sharktank-diarization\data\arzen")


def load_utt2spk(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def load_split_recordings(split_file: Path) -> set[str]:
    """Map split entries like 'C03-R04-S04' -> base recording ID 'C03-R04'."""
    recs: set[str] = set()
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token:
                continue
            parts = token.split("-")
            if len(parts) >= 2:
                recs.add(f"{parts[0]}-{parts[1]}")
    return recs


def main() -> None:
    asr_dir = ARZEN_ROOT / "ASR_files"
    rec_dir = ARZEN_ROOT / "recordings"
    splits_dir = ARZEN_ROOT / "splits"

    utt2spk = load_utt2spk(asr_dir / "utt2spk")

    # Group all segments by base recording ID (e.g. C03-R04)
    rec_segments: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    with (asr_dir / "segments").open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            utt_id, rec_spk, start_s, end_s = parts
            start, end = float(start_s), float(end_s)
            if end <= start:
                continue
            base_rec = "-".join(rec_spk.split("-")[:2])
            spk = utt2spk.get(utt_id, rec_spk.split("-")[-1])
            rec_segments[base_rec].append((start, end, spk))

    summary: dict[str, dict] = {}
    for split_name in ("train", "dev", "test"):
        split_recs = sorted(load_split_recordings(splits_dir / split_name))
        rttm_dir = OUT_ROOT / split_name / "rttm"
        rttm_dir.mkdir(parents=True, exist_ok=True)

        manifest: list[dict] = []
        total_segments = 0
        total_speech_sec = 0.0

        for rec_id in split_recs:
            wav_path = rec_dir / f"{rec_id}.WAV"
            if not wav_path.exists():
                wav_path = rec_dir / f"{rec_id}.wav"
            if not wav_path.exists():
                print(f"[WARN] Missing WAV for {rec_id}")
                continue

            segs = sorted(rec_segments.get(rec_id, []), key=lambda x: (x[0], x[1]))
            rttm_path = rttm_dir / f"{rec_id}.rttm"
            speakers = sorted({s for _, _, s in segs})

            with rttm_path.open("w", encoding="utf-8") as out:
                for start, end, spk in segs:
                    dur = end - start
                    total_speech_sec += dur
                    out.write(
                        f"SPEAKER {rec_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
                    )

            total_segments += len(segs)
            manifest.append(
                {
                    "recording_id": rec_id,
                    "audio_path": str(wav_path),
                    "rttm_path": str(rttm_path),
                    "num_speakers": len(speakers),
                    "speakers": speakers,
                    "num_segments": len(segs),
                }
            )

        manifest_path = OUT_ROOT / split_name / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary[split_name] = {
            "recordings": len(manifest),
            "segments": total_segments,
            "speech_minutes": round(total_speech_sec / 60.0, 2),
            "manifest": str(manifest_path),
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
