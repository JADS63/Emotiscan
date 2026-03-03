from __future__ import annotations

import os
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
        if model is None:
            return None, None, None
        p_audio_list = [
            predict_audio_probs(seg, sr, model, processor) for seg in segments
        ]
        vad_list = compute_vad_flags(segments, sr)
        return p_audio_list, vad_list, sr
    except Exception:
        return None, None, None


def _render_analyse_video(cfg, class_names: List[str], after_path: str) -> None:
    st.markdown('''
        <div style="text-align: center; padding: 1rem;">
            <h1 style="color: #FF4B4B;">Analyse Vidéo</h1>
            <p style="font-size: 1.2rem;">Téléversez une séquence vidéo pour analyser l'évolution des émotions image par image.</p>
        </div>
    ''', unsafe_allow_html=True)

    model_demo, device_demo, ckpt_demo, resolved_after = load_model_from_path(cfg, Path(after_path))
    if ckpt_demo is None:
        st.error("Impossible de charger les paramètres du système pour l'analyse vidéo.")
        return

    num_classes = len(class_names)
    hw = (int(cfg["data"]["img_height"]), int(cfg["data"]["img_width"]))

    vid_file = st.file_uploader("Téléversez une vidéo...", type=["mp4", "avi", "mov", "mkv"])

    if vid_file is None:
        st.info(f"👋 Analyse jusqu'à {MAX_VIDEO_SECONDS} secondes de vidéo.")
        return

    tmp_path = ROOT / "._tmp_video.mp4"
    with open(tmp_path, "wb") as f:
        f.write(vid_file.read())
    tmp_path = ROOT / "._tmp_video.mp4"
    with open(tmp_path, "wb") as f:
        f.write(vid_file.read())

    cap = cv2.VideoCapture(str(tmp_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), int(MAX_VIDEO_SECONDS * fps))
    infer_every = max(1, int(fps / INFERENCE_FPS))
    cap = cv2.VideoCapture(str(tmp_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), int(MAX_VIDEO_SECONDS * fps))
    infer_every = max(1, int(fps / INFERENCE_FPS))

    if st.button("Lancer mon analyse"):
        p_ema = np.ones(num_classes) / num_classes
        current_label = "neutral"
        records = []
        progress = st.progress(0)
        status_placeholder = st.empty()

        for i in range(max_frames):
            ok, frame = cap.read()
            if not ok: break

            if i % infer_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crop_rgb, face_bbox = auto_face_crop_rgb(rgb)

                if face_bbox is not None:
                    gray = to_grayscale_np(crop_rgb)
                    img_96 = np.array(Image.fromarray(gray).resize((hw[1], hw[0]), Image.BILINEAR), dtype=np.uint8)
                    x = preprocess_image(Image.fromarray(img_96), cfg)
                    img_96 = np.array(Image.fromarray(gray).resize((hw[1], hw[0]), Image.BILINEAR), dtype=np.uint8)
                    x = preprocess_image(Image.fromarray(img_96), cfg)
                    _, _, probs = predict_emotion(model_demo, x, device_demo, class_names)

                    p_ema = EMA_ALPHA * probs + (1 - EMA_ALPHA) * p_ema
                    current_label = class_names[int(np.argmax(p_ema))]
                    conf = float(np.max(p_ema))
                    records.append((i, True, current_label, conf))
                else:
                    records.append((i, False, "Pas de visage", 0.0))

            progress.progress(int((i + 1) / max_frames * 100))
            status_placeholder.caption(f"Analyse de la frame {i+1}/{max_frames}...")

        cap.release()
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        st.success("Analyse terminée ! Voici les résultats.")

        dfv = pd.DataFrame(records, columns=["frame", "detect", "émotion", "confiance"])
        emotion_counts = dfv[dfv["detect"]]["émotion"].value_counts().reindex(class_names, fill_value=0)

        c1, c2 = st.columns(2)
        with c1:
            st.write("### Émotion Dominante")
            dominant = emotion_counts.idxmax() if emotion_counts.sum() > 0 else "Aucune"
            st.subheader(dominant.capitalize())
            st.write("### Émotion Dominante")
            dominant = emotion_counts.idxmax() if emotion_counts.sum() > 0 else "Aucune"
            st.subheader(dominant.capitalize())
        with c2:
            st.write("### Distribution temporelle")
            st.bar_chart(emotion_counts)
    else:
        cap.release()
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _render_direct(cfg, class_names: List[str], after_path: str) -> None:
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

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start = st.button("▶️ Démarrer l'analyse", use_container_width=True)
    with col_btn2:
        stop = st.button("⏹️ Arrêter", use_container_width=True)

    if start:
        st.session_state.webcam_running = True
    if stop:
        st.session_state.webcam_running = False

    _, col_webcam, _ = st.columns([0.15, 0.7, 0.15])

    with col_webcam:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

    if st.session_state.get("webcam_running", False):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        p_ema = np.ones(num_classes) / num_classes

        while st.session_state.get("webcam_running", False):
            ok, frame = cap.read()
            if not ok:
                break

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

            overlay = _draw_overlay(rgb, face_bbox, display_label, display_conf)

            frame_placeholder.image(overlay, use_container_width=True)

            if face_bbox is not None:
                status_placeholder.markdown(
                    f"<div style='text-align: center;'><b>Émotion détectée : {display_label}</b> ({display_conf*100:.0f}%)</div>",
                    unsafe_allow_html=True
                )

            time.sleep(0.01)
        cap.release()


def render(cfg, class_names: List[str], after_path: str) -> None:
    mode = st.radio(
        "Mode",
        ["Analyse Vidéo (upload)", "Observation en direct"],
        horizontal=True,
        label_visibility="collapsed",
        key="video_tab_mode",
    )
    if mode == "Analyse Vidéo (upload)":
        _render_analyse_video(cfg, class_names, after_path)
    else:
        _render_direct(cfg, class_names, after_path)
