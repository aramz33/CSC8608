import os
import json
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps


@dataclass
class Segment:
    start_s: float
    end_s: float


def load_wav_mono_16k(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav.squeeze(0), sr


def main():
    in_path = "TP3/data/call_01.wav"
    out_path = "TP3/outputs/vad_segments_call_01.json"
    os.makedirs("TP3/outputs", exist_ok=True)

    wav, sr = load_wav_mono_16k(in_path)
    duration_s = wav.numel() / sr

    model = load_silero_vad()
    model.to("cpu").eval()

    speech_ts = get_speech_timestamps(
        wav.to(torch.float32),
        model,
        sampling_rate=16000,
    )

    segments: List[Segment] = []
    for seg in speech_ts:
        start_s = seg["start"] / sr
        end_s = seg["end"] / sr
        segments.append(Segment(start_s=start_s, end_s=end_s))

    min_dur_s = 0.30
    segments = [s for s in segments if (s.end_s - s.start_s) >= min_dur_s]

    total_speech_s = sum((s.end_s - s.start_s) for s in segments)
    speech_ratio = total_speech_s / max(duration_s, 1e-9)

    print("duration_s:", round(duration_s, 2))
    print("num_segments:", len(segments))
    print("total_speech_s:", round(total_speech_s, 2))
    print("speech_ratio:", round(speech_ratio, 3))

    payload = {
        "audio_path": in_path,
        "sample_rate": sr,
        "duration_s": duration_s,
        "min_segment_s": min_dur_s,
        "segments": [{"start_s": s.start_s, "end_s": s.end_s} for s in segments],
        "stats": {
            "num_segments": len(segments),
            "total_speech_s": total_speech_s,
            "speech_ratio": speech_ratio,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("saved:", out_path)


if __name__ == "__main__":
    main()
