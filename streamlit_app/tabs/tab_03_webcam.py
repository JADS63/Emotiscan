from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from streamlit_app.common import (
    ROOT,
    get_device,
    load_model_from_path,
    preprocess_image,
    predict_emotion,
    auto_face_crop_rgb,
    to_grayscale_np,
)
from streamlit_app.audio_utils import (
    extract_audio_mono_16k,
    segment_audio,
    load_audio_emotion_model,
    predict_audio_probs,
    audio_window_index,
    compute_vad_flags,
)

AUDIO_MODEL_ID = "prithivMLmods/Speech-Emotion-Classification"

MAX_VIDEO_SECONDS = 30
INFERENCE_FPS = 6
EMA_ALPHA = 0.25
CONFIDENCE_THRESHOLD = 0.55
GATING_K = 3
NO_FACE_RESET_SECONDS = 2.0


def _draw_overlay(frame_rgb: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], label: str, conf: float) -> np.ndarray:
    out = frame_rgb.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = f"{label} ({conf*100:.0f}%)" if label != "Pas de visage" else label
    cv2.putText(out, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(out, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)
    return out


def _update_ema_and_gating(
    p_ema: np.ndarray,
    p_new: np.ndarray,
    current_label: str,
    pending_label: str,
    pending_count: int,
    class_names: List[str],
    alpha: float,
    threshold: float,
    gating_k: int,
) -> Tuple[np.ndarray, str, str, int]:
    p_ema_new = alpha * p_new + (1 - alpha) * p_ema
    p_ema_new = p_ema_new / p_ema_new.sum()
    idx = int(np.argmax(p_ema_new))
    label_candidate = class_names[idx]
    conf = float(p_ema_new[idx])

    if conf < threshold:
        return p_ema_new, current_label, pending_label, pending_count

    if label_candidate == current_label:
        return p_ema_new, current_label, label_candidate, 0
    if label_candidate == pending_label:
        new_count = pending_count + 1
        if new_count >= gating_k:
            return p_ema_new, label_candidate, label_candidate, 0
        return p_ema_new, current_label, pending_label, new_count
    return p_ema_new, current_label, label_candidate, 1


@st.cache_data(show_spinner=False)
def _cached_audio_probs_and_vad(
    video_path: str,
    path_size: int,
    path_mtime: float,
    window_sec: float,
    hop_sec: float,
    model_id: str,
) -> Tuple[Optional[List[np.ndarray]], Optional[List[bool]], Optional[int]]:
    try:
        path = Path(video_path)
        if not path.is_file():
            return None, None, None
        audio, sr = extract_audio_mono_16k(path)
        if audio.size == 0:
            return None, None, None
        segments = segment_audio(audio, sr, window_sec, hop_sec)
        if not segments:
            return None, None, None
        device = get_device()
        model, processor, _ = load_audio_emotion_model(model_id, device)
        p_audio_list = [
            predict_audio_probs(seg, sr, model, processor) for seg in segments
        ]
        vad_list = compute_vad_flags(segments, sr)
        return p_audio_list, vad_list, sr
    except Exception:
        return None, None, None


def render(cfg, class_names: List[str], after_path: str) -> None:
    st.markdown('''
        <div style="text-align: center; padding: 1rem;">
            <h1 style="color: #FF4B4B;">Observation en direct</h1>
            <p style="font-size: 1.1rem;">Analyse en temps réel des émotions faciales.</p>
        </div>
    ''', unsafe_allow_html=True)

    model_demo, device_demo, ckpt_demo, _ = load_model_from_path(cfg, Path(after_path))
    if ckpt_demo is None:
        st.error("Impossible d'accéder à la caméra pour l'instant.")
        return

    num_classes = len(class_names)
    hw = (int(cfg["data"]["img_height"]), int(cfg["data"]["img_width"]))

    # Boutons de contrôle
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start = st.button("▶️ Démarrer l'analyse", use_container_width=True)
    with col_btn2:
        stop = st.button("⏹️ Arrêter", use_container_width=True)

    if start: st.session_state.webcam_running = True
    if stop: st.session_state.webcam_running = False

    # --- ASTUCE POUR RÉDUIRE LA TAILLE ---
    # On crée 3 colonnes, l'image sera dans celle du milieu (index 1)
    # [15% vide | 70% image | 15% vide]
    _, col_webcam, _ = st.columns([0.15, 0.7, 0.15])
    
    with col_webcam:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

    if st.session_state.get("webcam_running", False):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        p_ema = np.ones(num_classes) / num_classes
        
        while st.session_state.webcam_running:
            ok, frame = cap.read()
            if not ok: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop_rgb, face_bbox = auto_face_crop_rgb(rgb)
            
            display_label = "Je vous cherche..."
            display_conf = 0.0

            if face_bbox is not None:
                gray = to_grayscale_np(crop_rgb)
                img_96 = np.array(Image.fromarray(gray).resize((hw[1], hw[0]), Image.BILINEAR), dtype=np.uint8)
                x = preprocess_image(Image.fromarray(img_96), cfg)
                _, _, probs = predict_emotion(model_demo, x, device_demo, class_names)
                
                p_ema = EMA_ALPHA * probs + (1 - EMA_ALPHA) * p_ema
                display_label = class_names[int(np.argmax(p_ema))]
                display_conf = float(np.max(p_ema))

            # Dessin de l'overlay
            overlay = _draw_overlay(rgb, face_bbox, display_label, display_conf)
            
            # Affichage dans la colonne réduite
            frame_placeholder.image(overlay, use_container_width=True)
            
            if face_bbox is not None:
                status_placeholder.markdown(
                    f"<div style='text-align: center;'><b>Émotion détectée : {display_label}</b> ({display_conf*100:.0f}%)</div>", 
                    unsafe_allow_html=True
                )
            
            time.sleep(0.01)
        cap.release()
