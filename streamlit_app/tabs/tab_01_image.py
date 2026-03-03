from __future__ import annotations
import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import cv2
import streamlit as st
import torch

from streamlit_app.common import (
    load_model_from_path,
    preprocess_image,
    predict_emotion,
    auto_face_crop_rgb,
    to_grayscale_np,
)


def save_feedback(predicted, corrected, image_np):
    # --- CONFIGURATION DES CHEMINS ---
    base_data_dir = "data"
    csv_path = os.path.join(base_data_dir, "labels.csv")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    img_filename = f"user_fb_{timestamp}.jpg"
    rel_path = f"{corrected}/{img_filename}"
    full_img_dir = os.path.join(base_data_dir, "Train", corrected)
    full_img_path = os.path.join(full_img_dir, img_filename)

    os.makedirs(full_img_dir, exist_ok=True)
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(full_img_path, img_bgr)

    try:
        df = pd.read_csv(csv_path)
        new_row = {
            'Unnamed: 0': len(df),
            'pth': rel_path,
            'label': corrected,
            'relFCs': 1.0
        }

        # Ajout au DataFrame et sauvegarde
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        return True
    except Exception as e:
        print(f"Erreur lors de la mise à jour du CSV : {e}")
        return False


def render(cfg, class_names, after_path: str) -> None:
    # --- EN-TÊTE PERSONNIFIÉ ---
    st.markdown('''
        <div style="text-align: center; padding: 1rem;">
            <h1 style="color: #FF4B4B;">Analyse d'émotions faciales</h1>
            <p style="font-size: 1.2rem;">Le système au cœur d'EmotiScan.
            Confiez une image, et le système analysera l'émotion qui s'y dégage.</p>
        </div>
    ''', unsafe_allow_html=True)

    model_demo, device_demo, ckpt_demo, resolved_after = load_model_from_path(cfg, Path(after_path))
    if ckpt_demo is None:
        st.error(f"Impossible de charger les paramètres du système (modèle introuvable).")
        return

    uploaded = st.file_uploader("Téléversez une photo...", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded is None:
        st.info("👋 En attente d'une image pour commencer l'analyse !")
        return

    image_pil = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(image_pil)
    img_cropped, face_bbox = auto_face_crop_rgb(img_rgb)
    face_detected = face_bbox is not None
    img_gray = to_grayscale_np(img_cropped)
    h_target, w_target = int(cfg["data"]["img_height"]), int(cfg["data"]["img_width"])
    img_96x96_gray = np.array(Image.fromarray(img_gray).resize((w_target, h_target), Image.BILINEAR), dtype=np.uint8)

    # Inférence
    x = preprocess_image(Image.fromarray(img_96x96_gray), cfg)
    pred_name, pred_conf, probs = predict_emotion(model_demo, x, device_demo, class_names)

    # Image avec cadre
    display_original = img_rgb.copy()
    if face_detected:
        x_b, y_b, w_b, h_b = face_bbox
        cv2.rectangle(display_original, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 255, 0), 3)

    if not face_detected:
        st.warning(
            "⚠️ Aucun visage détecté par OpenCV : l'image entière est utilisée pour l'analyse. "
            "La précision peut en pâtir. Pour de meilleurs résultats, utilisez une photo où le visage est bien visible et de face."
        )
    st.write("### Analyse en cours...")
    col_img_left, col_img_right = st.columns(2)
    with col_img_left:
        st.caption("Image originale :" + (" (rectangle vert = visage détecté)" if face_detected else ""))
        st.image(display_original, use_container_width=True)
    with col_img_right:
        st.caption(
            "Traitement de l'image : "
            + ("visage recadré puis" if face_detected else "image entière convertie en")
            + " niveaux de gris, redimensionné 96×96"
        )
        st.image(img_96x96_gray, use_container_width=True)

    st.divider()

    # --- RÉSULTATS ET FEEDBACK PERSONNIFIÉS ---
    col_result, col_feedback = st.columns(2)

    with col_result:
        st.write(f"### Résultat")
        st.write(f"Émotion détectée : **{pred_name}**.")
        st.write(f"Niveau de confiance : **{pred_conf*100:.2f}%**.")
        st.progress(float(pred_conf))

    with col_feedback:
        st.write("### Feedback")
        st.write("La détection est-elle correcte ?")
        c_vote1, c_vote2 = st.columns(2)

        with c_vote1:
            if st.button("👍 Oui, correct !", use_container_width=True, key="btn_correct"):
                st.session_state.show_correction = False
                st.balloons()
                save_feedback(pred_name, pred_name, img_cropped)
                st.success("Merci pour votre retour !")

        with c_vote2:
            if st.button("👎 Non, incorrect", use_container_width=True, key="btn_incorrect"):
                st.session_state.show_correction = True

        if st.session_state.get("show_correction", False):
            st.info("Quelle émotion aurait dû être détectée ?")
            true_class = st.selectbox("Choisissez l'émotion réelle :", list(class_names))
            if st.button("Valider la correction"):
                save_feedback(pred_name, true_class, img_cropped)
                st.snow()
                st.success(f"Correction enregistrée : '{true_class}'.")
                st.session_state.show_correction = False

    # --- STATS ---
    st.markdown(f'<div class="section-title">Détail des probabilités par classe</div>', unsafe_allow_html=True)
    df_probs = pd.DataFrame({"classe": class_names, "probabilité (%)": (probs * 100)})
    df_probs = df_probs.sort_values("probabilité (%)", ascending=False)
    st.bar_chart(df_probs.set_index("classe")["probabilité (%)"], height=300)

    # --- LIME : Pourquoi le système a choisi ça ---
    st.divider()
    st.markdown("### Explication de la détection (LIME)")
    st.caption(
        "LIME met en évidence les zones de l’image qui influencent la détection vers la classe prédite (vert) "
        "ou au contraire la freinent (rouge). [En savoir plus](https://christophm.github.io/interpretable-ml-book/lime.html)"
    )
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries

        mean = float(cfg["data"]["mean"])
        std = float(cfg["data"]["std"])
        num_classes = len(class_names)

        def _predict_fn(images_batch: np.ndarray) -> np.ndarray:
            if images_batch.ndim == 4:
                gray = images_batch.mean(axis=-1)
            else:
                gray = images_batch
            gray = gray.astype(np.float64)
            if gray.max() > 1.5:
                gray = gray / 255.0
            x = torch.from_numpy(gray[:, np.newaxis, :, :]).float()
            x = (x - mean) / std
            x = x.to(device_demo)
            with torch.no_grad():
                logits = model_demo(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        img_for_lime = img_96x96_gray.astype(np.float64) / 255.0
        img_for_lime = np.stack([img_for_lime, img_for_lime, img_for_lime], axis=-1)

        explainer = lime_image.LimeImageExplainer(random_state=42)
        explanation = explainer.explain_instance(
            img_for_lime,
            _predict_fn,
            labels=(pred_idx := list(class_names).index(pred_name),),
            hide_color=0,
            num_samples=200,
            batch_size=32,
        )
        temp, mask = explanation.get_image_and_mask(
            pred_idx,
            positive_only=False,
            num_features=6,
            hide_rest=False,
        )
        overlay = mark_boundaries(temp, mask, color=(0, 1, 0), mode="thick")
        overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

        col_lime1, col_lime2 = st.columns(2)
        with col_lime1:
            st.image(img_96x96_gray, caption="Image analysée (niveaux de gris)", use_container_width=True)
        with col_lime2:
            st.image(overlay, caption="Explication LIME : vert = pour cette émotion, rouge = contre", use_container_width=True)
    except ImportError:
        st.info("Pour activer les explications LIME, installez : `pip install lime scikit-image`")
    except Exception as e:
        st.warning(f"Explication LIME non disponible : {e}")
