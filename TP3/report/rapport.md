# TP3 – Deep learning pour l'audio

## Dépôt

Lien : https://github.com/adamramsis/CSC8608

## Environnement d'exécution

| Librairie | Version |
|---|---|
| torch | 2.11.0 |
| torchaudio | 2.11.0 |
| transformers | 5.5.4 |
| datasets | 4.8.4 |
| device | MPS (Apple M3) |

```
=== TP3 sanity check ===
torch: 2.11.0
torchaudio: 2.11.0
transformers: 5.5.4
datasets: 4.8.4
device: mps
wav_shape: (1, 16000)
logmel_shape: (1, 80, 101)
```

## Arborescence TP3/

```
TP3/
  assets/
  data/
    call_01.wav
  outputs/
    vad_segments_call_01.json
    asr_call_01.json
    call_summary_call_01.json
    tts_reply_call_01.wav
    pipeline_summary_call_01.json
  report/
    rapport.md
    screenshots/
  sanity_check.py
  inspect_audio.py
  vad_segment.py
  asr_whisper.py
  callcenter_analytics.py
  tts_reply.py
  asr_tts_check.py
  run_pipeline.py
  requirements.txt
```

---

## Exercice 1 – Initialisation du TP3 et vérification de l'environnement

Le script `sanity_check.py` vérifie les versions des dépendances, détecte le device disponible (MPS sur M3), génère un signal sinusoïdal à 440 Hz et calcule le Log-Mel spectrogram correspondant.

Sortie obtenue :

```
=== TP3 sanity check ===
torch: 2.11.0  |  torchaudio: 2.11.0  |  transformers: 5.5.4  |  datasets: 4.8.4
device: mps
wav_shape: (1, 16000)
logmel_shape: (1, 80, 101)
```

Le device MPS est disponible (Apple Silicon M3). Le Log-Mel de shape `(1, 80, 101)` correspond à 80 filtres Mel sur ~1 seconde avec `hop_length=160` (101 frames × 160 / 16000 ≈ 1.01s).

> Note : `torch.cuda.is_available()` retourne `False` sur MacBook M3. MPS est l'accélérateur matériel disponible localement.

---

## Exercice 2 – Constituer un mini-jeu de données

L'enregistrement `call_01.wav` simule un appel client signalant une commande endommagée, incluant des éléments PII (numéro de commande, email épelé, numéro de téléphone). Le fichier est en WAV mono 16 kHz.

### Métadonnées audio

```
path: TP3/data/call_01.wav
sr: 16000
shape: (1, 629029)
duration_s: 39.31
rms: 0.0298
clipping_rate: 0.0
```

La durée est de 39 secondes (lecture fluide, sans pauses excessives). Le RMS de 0.03 indique un niveau d'enregistrement correct, et le taux de clipping nul (`clipping_rate: 0.0`) confirme l'absence de saturation micro.

---

## Exercice 3 – VAD : segmenter la parole

Le VAD utilisé est **Silero-VAD** (`silero-vad` pip package, API `load_silero_vad()`), modèle léger pré-entraîné optimisé pour la détection voix/silence sur audio 16 kHz.

### Résultats VAD

```
duration_s: 39.31
num_segments: 9
total_speech_s: 29.56
speech_ratio: 0.752
saved: TP3/outputs/vad_segments_call_01.json
```

### Extrait JSON (5 premiers segments)

```json
[
  {"start_s": 2.40,  "end_s": 7.13},
  {"start_s": 7.46,  "end_s": 13.66},
  {"start_s": 14.02, "end_s": 17.69},
  {"start_s": 18.88, "end_s": 23.81},
  {"start_s": 24.26, "end_s": 27.20}
]
```

### Analyse speech/silence

Le ratio speech de 0.752 est cohérent avec une lecture continue à voix claire : environ 25% de silence correspond aux pauses respiratoires naturelles entre phrases. Les 9 segments reflètent bien la structure sémantique du texte (salutation, problème, demande, données PII, remerciement). On observe deux micro-segments (seg 5 : 0.5s, seg 8 : 0.5s) correspondant à des pauses d'hésitation lors de la prononciation de l'email.

### Comparaison min_dur_s = 0.30 vs 0.60

| `min_dur_s` | `num_segments` | `speech_ratio` |
|---|---|---|
| 0.30 | 9 | 0.752 |
| 0.60 | 7 | 0.727 |

En passant de 0.30 à 0.60, deux micro-segments disparaissent (`That's...` et `Thank you.`), réduisant légèrement le ratio. Le seuil 0.30 est préférable ici pour ne pas perdre de PII contenues dans des fragments courts.

---

## Exercice 4 – ASR avec Whisper

### Métriques

```
model_id: openai/whisper-base
device: mps
audio_duration_s: 39.31
elapsed_s: 4.45
rtf: 0.113
saved: TP3/outputs/asr_call_01.json
```

RTF de 0.113 : Whisper-base traite l'audio ~9× plus vite que le temps réel sur CPU (MPS non utilisé par le pipeline HuggingFace). Très en-dessous du budget de 5 minutes.

### Extrait segments (5 premiers)

```json
[
  {"segment_id": 0, "start_s": 2.40,  "end_s": 7.13,  "text": "Hello, thank you for calling customer support. My name is Alex and I will help you today."},
  {"segment_id": 1, "start_s": 7.46,  "end_s": 13.66, "text": "I'm calling about an order that arrived damaged. The package was delivered yesterday, but the screen is cracked."},
  {"segment_id": 2, "start_s": 14.02, "end_s": 17.69, "text": "I would like a refund or replacement as soon as possible."},
  {"segment_id": 3, "start_s": 18.88, "end_s": 23.81, "text": "The order number is AX19735."},
  {"segment_id": 4, "start_s": 24.26, "end_s": 27.20, "text": "You can reach me at john.smith."}
]
```

### Extrait full_text

```
Hello, thank you for calling customer support. My name is Alex and I will help you today.
I'm calling about an order that arrived damaged. The package was delivered yesterday, but the screen is cracked.
I would like a refund or replacement as soon as possible. The order number is AX19735.
You can reach me at john.smith. That's... Example.com. Also my phone number is 5550199. Thank you.
```

### Analyse VAD → ASR

La segmentation VAD aide globalement : elle évite que Whisper dérive sur les silences et produit des transcriptions propres par phrase. Cependant, la coupure de l'email en trois segments distincts (seg 4 : `You can reach me at john.smith.`, seg 5 : `That's...`, seg 6 : `Example.com.`) est problématique. Whisper ne peut pas reconstruire `john.smith@example.com` car il traite chaque segment indépendamment. Un merge des segments proches (gap < 0.5s) avant l'ASR préserverait les entités longues comme les emails épelés.

---

## Exercice 5 – Call center analytics : redaction PII + intention

### Résultats analytics

```
intent: delivery_issue
pii_stats: {'emails': 0, 'phones': 0, 'orders': 1}
top_terms: [('order', 3), ('thank', 2), ('calling', 2), ('number', 2), ('hello', 1)]
saved: TP3/outputs/call_summary_call_01.json
```

### Extrait call_summary_call_01.json

```json
{
  "pii_stats": {"emails": 0, "phones": 0, "orders": 1},
  "intent_scores": {
    "refund_or_replacement": 4,
    "delivery_issue": 7,
    "general_support": 6
  },
  "intent": "delivery_issue",
  "top_terms": [["order", 3], ["thank", 2], ["calling", 2], ["number", 2], ["hello", 1]]
}
```

Extrait `redacted_text` (3 phrases) :

```
...the order number is [REDACTED_ORDER]@john.smith.thats example.com.
also my phone number is 5550199.thank you.
```

### Comparaison avant/après post-traitement PII

Le post-traitement amélioré (bloc `normalize_spelled_tokens` + `redact_order_id`) détecte et masque correctement le numéro de commande AX19735 → `[REDACTED_ORDER]`. En revanche, l'email et le téléphone restent non redactés. L'email `john.smith@example.com` a été épelé en trois segments VAD distincts → le `full_text` contient `john.smith. That's... Example.com.` avec ponctuation, que le regex email standard ne peut pas matcher. Le numéro `5550199` apparaît dans le texte normalisé mais collé à `@john.smith` et `thank`, empêchant le regex `PHONE_RE` (7+ digits consécutifs) de l'isoler.

### Réflexion : impact des erreurs Whisper

L'erreur la plus impactante est la fragmentation de l'email : Whisper transcrit correctement chaque segment mais la ponctuation injectée (`john.smith.` avec point final) et la séparation en 3 segments empêchent toute détection PII regex. L'intent `delivery_issue` est sélectionné (score 7) au lieu de `refund_or_replacement` (score 4) car les mots de livraison (`delivered`, `package`, `arrived`, `order`) sont plus fréquents que les mots de remboursement dans le transcript, bien que la demande explicite soit un refund. En production, une erreur d'intent de ce type peut provoquer un mauvais routage de l'appel (service livraison vs service remboursement).

---

## Exercice 6 – TTS léger

### Métriques TTS

```
tts_model_id: facebook/mms-tts-eng
device: cpu
audio_dur_s: 8.38
elapsed_s: 0.95
rtf: 0.114
saved: TP3/outputs/tts_reply_call_01.wav
```

### Métadonnées WAV généré

```
Duration: 00:00:08.38
Audio: pcm_s16le, 16000 Hz, 1 channels (mono), 256 kb/s
```

### Observation qualité TTS

`facebook/mms-tts-eng` produit une voix intelligible avec une prosodie correcte mais légèrement monotone (pas d'intonation interrogative sur "Please confirm your preferred option"). Aucun artefact métallique notable. Le RTF de 0.114 est excellent : la synthèse est ~9× plus rapide que le temps réel, compatible avec une utilisation en temps réel en production. La qualité est suffisante pour un prototype call center.

### ASR sur WAV TTS (vérification intelligibilité)

```
model_id: openai/whisper-base
elapsed_s: 1.55
text: Thanks for calling. I am sorry your order arrived, damaged.
      I can offer a replacement or refund. Please confirm your preferred option.
```

Whisper retranscrit le WAV TTS avec une fidélité quasi-parfaite au texte source (seul "a replacement" devient "replacement", virgule ajoutée). Intelligibilité confirmée.

---

## Exercice 7 – Pipeline end-to-end

### Pipeline summary

```
=== PIPELINE SUMMARY ===
audio_path: TP3/data/call_01.wav
duration_s: 39.31
num_segments: 9
speech_ratio: 0.752
asr_model: openai/whisper-base
asr_device: mps
asr_rtf: 0.113
intent: delivery_issue
pii_stats: {'emails': 0, 'phones': 0, 'orders': 1}
tts_generated: True
```

### pipeline_summary_call_01.json

```json
{
  "audio_path": "TP3/data/call_01.wav",
  "duration_s": 39.3143125,
  "num_segments": 9,
  "speech_ratio": 0.7519907667214071,
  "asr_model": "openai/whisper-base",
  "asr_device": "mps",
  "asr_rtf": 0.11330194069179007,
  "intent": "delivery_issue",
  "pii_stats": {"emails": 0, "phones": 0, "orders": 1},
  "tts_generated": true
}
```

### Engineering note

**Goulet d'étranglement (temps) :** L'ASR Whisper représente 4.45s sur 39s d'audio (RTF 0.113). Le traitement segment-par-segment (9 appels séquentiels au pipeline) génère un overhead de chargement de contexte à chaque itération. Un batch de tous les segments en un seul appel réduirait ce coût de ~30%.

**Étape la plus fragile (qualité) :** La redaction PII est l'étape la plus fragile. Elle dépend de la qualité de transcription ASR et de la segmentation VAD. L'email épelé sur plusieurs segments VAD n'est pas reconstruit, et le numéro de téléphone collé à d'autres tokens après normalisation échappe au regex. Une erreur PII non détectée en production est un risque RGPD direct.

**Deux améliorations concrètes pour industrialiser :**
1. **Merge des segments VAD proches** (gap < 0.5s) avant l'ASR : préserve la cohérence des entités longues (email, numéro de commande épelé) et réduit le nombre d'appels au pipeline.
2. **Fenêtre glissante inter-segments pour la redaction PII** : concaténer le texte de deux segments consécutifs avant d'appliquer les regex, pour capturer les entités qui franchissent une coupure VAD.

---

## Réflexion finale

Ce TP illustre les limites d'un pipeline purement heuristique sur de la parole spontanée. La segmentation VAD, bien qu'efficace pour isoler la parole du silence, introduit des coupures qui cassent les entités sémantiques (emails, numéros). Whisper-base offre une transcription de bonne qualité pour un modèle léger (RTF 0.11 sur CPU), mais ne gère pas les apostrophes et ponctuation de manière homogène selon les contextes. Pour industrialiser : normalisation post-ASR systématique, merge VAD adaptatif, et remplacement des regex PII par un modèle NER spécialisé (ex. `dslim/bert-base-NER`) pour les entités nommées épelées ou fragmentées.
