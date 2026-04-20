import torch
import torchaudio


def rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x**2)).item())


def clipping_rate(x: torch.Tensor, thr: float = 0.99) -> float:
    return float((x.abs() > thr).float().mean().item())


def main():
    path = "TP3/data/call_01.wav"
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)  # force mono [1, time]
    num_samples = wav.shape[1]
    duration_s = num_samples / sr

    print("path:", path)
    print("sr:", sr)
    print("shape:", tuple(wav.shape))
    print("duration_s:", round(duration_s, 2))
    print("rms:", round(rms(wav), 4))
    print("clipping_rate:", round(clipping_rate(wav), 4))


if __name__ == "__main__":
    main()
