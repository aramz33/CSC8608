# TP1 – Segmentation interactive avec SAM

Segmentation d'images interactive via Segment Anything Model (SAM) de Meta, avec interface Streamlit.

## Lancement

```bash
source .venv/bin/activate
streamlit run TP1/src/app.py --server.port 8501
```

## Structure

```
TP1/
├── data/images/        # Images d'entrée
├── models/             # Checkpoint SAM (.pth) — non commité
├── outputs/overlays/   # Overlays générés
├── report/             # Rapport et captures
└── src/                # Code source (app, sam_utils, geom_utils, viz_utils)
```

## Dépendances

```bash
pip install -r TP1/requirements.txt
```
