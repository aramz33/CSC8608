import time
import torch
from transformers import pipeline


def main():
    wav_path = "TP3/outputs/tts_reply_call_01.wav"
    model_id = "openai/whisper-base"

    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=device,
    )

    t0 = time.time()
    out = asr(wav_path, generate_kwargs={"language": "english"})
    t1 = time.time()

    print("model_id:", model_id)
    print("elapsed_s:", round(t1 - t0, 2))
    print("text:", out.get("text", "").strip())


if __name__ == "__main__":
    main()
