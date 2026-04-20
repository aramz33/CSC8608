import os
import json
import time
from typing import Dict, Any, List

import torch
import torchaudio
from transformers import pipeline


def load_wav_mono_16k(path: str):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav.squeeze(0), sr


def main():
    audio_path = "TP3/data/call_01.wav"
    vad_path = "TP3/outputs/vad_segments_call_01.json"
    out_path = "TP3/outputs/asr_call_01.json"
    os.makedirs("TP3/outputs", exist_ok=True)

    wav, sr = load_wav_mono_16k(audio_path)
    audio_duration_s = wav.numel() / sr

    with open(vad_path, "r", encoding="utf-8") as f:
        vad_payload = json.load(f)
    segments = vad_payload["segments"]

    model_id = "openai/whisper-base"

    # MPS for M3; fall back to CPU (no CUDA on MacBook)
    if torch.backends.mps.is_available():
        device = "mps"
        pipeline_device = 0 if torch.cuda.is_available() else -1
    else:
        device = "cpu"
        pipeline_device = -1

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=pipeline_device,
    )

    t0 = time.time()
    results: List[Dict[str, Any]] = []

    for i, seg in enumerate(segments):
        start_s = float(seg["start_s"])
        end_s = float(seg["end_s"])
        start = int(start_s * sr)
        end = int(end_s * sr)
        seg_wav = wav[start:end]

        inp = {"array": seg_wav.numpy(), "sampling_rate": sr}
        out = asr(inp, generate_kwargs={"language": "english"})
        text = out.get("text", "").strip()

        results.append({
            "segment_id": i,
            "start_s": start_s,
            "end_s": end_s,
            "text": text,
        })

    t1 = time.time()
    elapsed_s = t1 - t0
    rtf = elapsed_s / max(audio_duration_s, 1e-9)

    full_text = " ".join([r["text"] for r in results]).strip()

    payload = {
        "audio_path": audio_path,
        "model_id": model_id,
        "device": device,
        "audio_duration_s": audio_duration_s,
        "elapsed_s": elapsed_s,
        "rtf": rtf,
        "segments": results,
        "full_text": full_text,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("model_id:", model_id)
    print("device:", device)
    print("audio_duration_s:", round(audio_duration_s, 2))
    print("elapsed_s:", round(elapsed_s, 2))
    print("rtf:", round(rtf, 3))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
