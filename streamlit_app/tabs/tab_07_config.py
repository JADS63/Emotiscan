"""
Tab 7: Configuration
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any


def render(cfg: Dict[str, Any], before_path: str, after_path: str):
    st.markdown("### Configuration")
    
    # Configuration du modele
    st.markdown("#### Modele")
    col1, col2 = st.columns(2)
    
    with col1:
        st.json({
            "name": cfg["model"]["name"],
            "dropout": cfg["model"]["dropout"],
            "pretrained": cfg["model"].get("pretrained", True),
            "freeze_layers": cfg["model"].get("freeze_layers", 0),
            "hidden_size": cfg["model"].get("hidden_size", 0),
        })
    
    with col2:
        st.markdown("**Description:**")
        st.write(f"- Architecture: **ResNet-18**")
        st.write(f"- Dropout: **{cfg['model']['dropout']:.2%}**")
        freeze = cfg["model"].get("freeze_layers", 0)
        if freeze > 0:
            st.write(f"- Couches gelees: **{freeze}** (sur 4)")
        else:
            st.write("- Toutes les couches sont entrainables")
    
    # Configuration des donnees
    st.markdown("#### Donnees")
    st.json({
        "img_height": cfg["data"]["img_height"],
        "img_width": cfg["data"]["img_width"],
        "channels": cfg["data"]["channels"],
        "num_classes": cfg["data"]["num_classes"],
        "class_names": cfg["data"]["class_names"],
        "mean": cfg["data"]["mean"],
        "std": cfg["data"]["std"],
    })
    
    # Configuration de l'entrainement
    st.markdown("#### Entrainement")
    st.json({
        "batch_size": cfg["train"]["batch_size"],
        "epochs": cfg["train"]["epochs"],
        "learning_rate": cfg["train"]["learning_rate"],
        "weight_decay": cfg["train"]["weight_decay"],
        "optimizer": cfg["train"]["optimizer"],
        "scheduler": cfg["train"]["scheduler"],
        "early_stopping_patience": cfg["train"]["early_stopping_patience"],
        "label_smoothing": cfg["train"].get("label_smoothing", 0.1),
    })
    
    # Augmentation
    st.markdown("#### Augmentation")
    aug_config = {
        "horizontal_flip": cfg["data"].get("hflip_p", 0.5),
        "rotation_degrees": cfg["data"].get("aug_rotation_deg", 15),
        "color_jitter": cfg["data"].get("aug_color_jitter", 0.2),
        "random_erasing": cfg["data"].get("aug_random_erasing", 0.15),
        "affine_scale": cfg["data"].get("aug_affine_scale", [0.9, 1.1]),
    }
    st.json(aug_config)
    
    # Chemins des checkpoints
    st.markdown("#### Checkpoints")
    st.write(f"- **AVANT**: `{before_path}`")
    st.write(f"- **APRES**: `{after_path}`")
    
    # Verification des fichiers
    ROOT = Path(__file__).parents[2]
    before_exists = (ROOT / before_path).exists() if before_path else False
    after_exists = (ROOT / after_path).exists() if after_path else False
    
    if before_exists and after_exists:
        st.success("Les deux fichiers de poids existent.")
    elif after_exists:
        st.warning("Le fichier AVANT n'existe pas. Lancez l'entrainement sans optimisation d'abord.")
    else:
        st.error("Fichiers de poids manquants. Lancez `python train.py` pour entrainer le modele.")
    
    # Commandes utiles
    st.markdown("#### Commandes utiles")
    st.code("""
# Entrainer le modele
python train.py

# Entrainer avec recherche d'hyperparametres
python train.py --tune --trials 30


# Recherche d'hyperparametres avancee
python scripts/tune_optuna.py --trials 50

# Valider le modele
python scripts/validate_model.py
    """, language="bash")
