# Screenshots à faire avant le rendu

Chaque screenshot = terminal macOS avec la commande visible + la sortie complète.
Sauvegarder dans `TP3/report/screenshots/` avec le nom exact indiqué.

---

## Ex1 – Sanity check

**Fichier :** `screenshots/sanity_check.png`

```bash
source .venv/bin/activate && python TP3/sanity_check.py
```

Doit montrer : torch version, torchaudio, transformers, datasets, device, wav_shape, logmel_shape.

---

## Ex2 – Métadonnées audio (ffprobe)

**Fichier :** `screenshots/ffprobe_call01.png`

```bash
ffprobe TP3/data/call_01.wav
```

Doit montrer : durée, sample rate (16000 Hz), canaux (mono).

---

## Ex2 – Inspect audio

**Fichier :** `screenshots/inspect_audio.png`

```bash
source .venv/bin/activate && python TP3/inspect_audio.py
```

Doit montrer : path, sr, shape, duration_s, rms, clipping_rate.

---

## Ex3 – VAD stats

**Fichier :** `screenshots/vad_stats.png`

```bash
source .venv/bin/activate && python TP3/vad_segment.py
```

Doit montrer : duration_s, num_segments, total_speech_s, speech_ratio, "saved: ..."

---

## Ex4 – ASR stats

**Fichier :** `screenshots/asr_stats.png`

```bash
source .venv/bin/activate && python TP3/asr_whisper.py
```

Doit montrer : model_id, device, audio_duration_s, elapsed_s, rtf, "saved: ..."

---

## Ex5 – Analytics (intent + PII)

**Fichier :** `screenshots/analytics_stats.png`

```bash
source .venv/bin/activate && python TP3/callcenter_analytics.py
```

Doit montrer : intent, pii_stats, top_terms[:5], "saved: ..."

---

## Ex6 – TTS stats

**Fichier :** `screenshots/tts_stats.png`

```bash
source .venv/bin/activate && python TP3/tts_reply.py
```

Doit montrer : tts_model_id, device, audio_dur_s, elapsed_s, rtf, "saved: ..."

---

## Ex6 – Métadonnées WAV TTS (ffprobe)

**Fichier :** `screenshots/tts_ffprobe.png`

```bash
ffprobe TP3/outputs/tts_reply_call_01.wav
```

Doit montrer : durée, sample rate, canaux.

---

## Ex7 – Pipeline summary

**Fichier :** `screenshots/pipeline_summary.png`

```bash
source .venv/bin/activate && python TP3/run_pipeline.py
```

Doit montrer : le bloc "=== PIPELINE SUMMARY ===" complet avec toutes les clés.

---

## Après les screenshots

1. Placer tous les PNG dans `TP3/report/screenshots/`
2. Dans `rapport.md`, remplacer les blocs de texte des sections "Métriques" par les images :
   ```markdown
   ![sanity_check](screenshots/sanity_check.png)
   ```
3. `git add TP3/report/screenshots/*.png`
4. `git commit -m "TP3: add terminal screenshots to report"`
5. `git push`
