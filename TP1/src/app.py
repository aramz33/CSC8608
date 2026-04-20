import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from sam_utils import load_sam_predictor, predict_mask_from_box, predict_masks_from_box_and_points
from geom_utils import mask_area, mask_bbox, mask_perimeter
from viz_utils import render_overlay


DATA_DIR = Path(__file__).parent.parent / "data" / "images"
OUT_DIR = Path(__file__).parent.parent / "outputs" / "overlays"
CKPT_PATH = str(Path(__file__).parent.parent / "models" / "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"


def load_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Image illisible: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_box_preview(image_rgb: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    preview = image_rgb.copy()
    bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = [int(v) for v in box_xyxy.tolist()]
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@st.cache_resource
def get_predictor():
    return load_sam_predictor(CKPT_PATH, model_type=MODEL_TYPE)


st.set_page_config(page_title="TP1 - SAM Segmentation", layout="wide")
st.title("TP1 — Segmentation interactive (SAM)")

if "points" not in st.session_state:
    st.session_state["points"] = []
if "last_pred" not in st.session_state:
    st.session_state["last_pred"] = None

# 1) Image selection
imgs = sorted([p for p in DATA_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
if len(imgs) == 0:
    st.error("Aucune image trouvée dans TP1/data/images/")
    st.stop()

img_name = st.selectbox("Choisir une image", [p.name for p in imgs])
img_path = DATA_DIR / img_name
img = load_image_rgb(img_path)
H, W = img.shape[:2]

# 2) Bbox sliders
st.subheader("Bounding box (pixels)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    x1 = st.slider("x1", 0, W - 1, W // 4)
with col2:
    y1 = st.slider("y1", 0, H - 1, H // 4)
with col3:
    x2 = st.slider("x2", 0, W - 1, 3 * W // 4)
with col4:
    y2 = st.slider("y2", 0, H - 1, 3 * H // 4)

x_min, x_max = (x1, x2) if x1 < x2 else (x2, x1)
y_min, y_max = (y1, y2) if y1 < y2 else (y2, y1)
box = np.array([x_min, y_min, x_max, y_max], dtype=np.int32)

if (x_max - x_min) < 10 or (y_max - y_min) < 10:
    st.warning("BBox très petite : essayez une bbox plus large.")

# 3) Live bbox preview
preview = draw_box_preview(img, box)
st.image(preview, caption="Prévisualisation : bbox (avant segmentation)", use_container_width=True)

# 4) Points FG/BG (Ex6)
with st.expander("Points de guidage FG/BG (optionnel)"):
    st.caption("Format : x,y — un point par ligne. FG = foreground (inclure), BG = background (exclure).")
    col_fg, col_bg = st.columns(2)
    with col_fg:
        fg_text = st.text_area("Points FG (foreground)", placeholder="100,200\n150,300")
    with col_bg:
        bg_text = st.text_area("Points BG (background)", placeholder="10,10\n20,30")

    def parse_points(text: str):
        pts = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    pts.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    pass
        return pts

    fg_pts = parse_points(fg_text)
    bg_pts = parse_points(bg_text)

    use_points = len(fg_pts) > 0 or len(bg_pts) > 0
    if use_points:
        all_pts = fg_pts + bg_pts
        all_lbl = [1] * len(fg_pts) + [0] * len(bg_pts)
        point_coords = np.array(all_pts, dtype=np.float32)
        point_labels = np.array(all_lbl, dtype=np.int64)
    else:
        point_coords, point_labels = None, None

# 5) Segmentation
do_segment = st.button("Segmenter")
if do_segment:
    predictor = get_predictor()
    t0 = time.time()

    if use_points:
        masks, scores = predict_masks_from_box_and_points(
            predictor, img, box, point_coords, point_labels, multimask=True
        )
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        score = float(scores[best_idx])
    else:
        mask, score = predict_mask_from_box(predictor, img, box, multimask=True)

    dt = (time.time() - t0) * 1000.0

    overlay = render_overlay(img, mask, box, alpha=0.5)
    m_area = mask_area(mask)
    m_bbox = mask_bbox(mask)
    m_per = mask_perimeter(mask)

    st.session_state["last_pred"] = {
        "overlay": overlay,
        "mask": mask,
        "score": score,
        "dt": dt,
        "area": m_area,
        "bbox": m_bbox,
        "perimeter": m_per,
        "img_stem": img_path.stem,
    }

if st.session_state["last_pred"] is not None:
    p = st.session_state["last_pred"]

    st.subheader("Résultat")
    st.image(p["overlay"], caption=f"score={p['score']:.3f} | time={p['dt']:.1f} ms", use_container_width=True)
    st.write({
        "score": round(float(p["score"]), 4),
        "time_ms": round(float(p["dt"]), 1),
        "area_px": int(p["area"]),
        "mask_bbox": p["bbox"],
        "perimeter": round(float(p["perimeter"]), 1),
    })

    if st.button("Sauvegarder overlay"):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"overlay_{p['img_stem']}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(p["overlay"], cv2.COLOR_RGB2BGR))
        st.success(f"Sauvegardé : {out_path}")
