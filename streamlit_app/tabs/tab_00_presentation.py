"""
Tab 0: Presentation du projet EmotiScan
"""

import streamlit as st
from typing import Dict, Any


def render(cfg: Dict[str, Any]):
    st.markdown("### Presentation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **EmotiScan** est un systeme de reconnaissance d'emotions faciales base sur 
        le traitement d'images et l'analyse de patterns.
        
        #### Architecture
        - **Modele**: ResNet-18 adapte pour images en niveaux de gris
        - **Dataset**: AffectNet (7 emotions)
        - **Optimisation**: Recherche d'hyperparametres avec Optuna
        
        #### Emotions detectees
        1. Colere (anger)
        2. Degout (disgust)
        3. Peur (fear)
        4. Joie (happy)
        5. Neutre (neutral)
        6. Tristesse (sad)
        7. Surprise (surprise)
        
        #### Fonctionnalites
        - Detection en temps reel sur images
        - Detection sur flux video/webcam
        - Comparaison avant/apres optimisation
        - Visualisation des metriques
        """)
    
    with col2:
        st.markdown("#### Configuration actuelle")
        st.json({
            "modele": cfg["model"]["name"],
            "image_size": f"{cfg['data']['img_height']}x{cfg['data']['img_width']}",
            "classes": cfg["data"]["num_classes"],
            "dropout": cfg["model"]["dropout"],
            "pretrained": cfg["model"].get("pretrained", True),
        })
        
        st.markdown("#### Version")
        st.info(f"v{cfg['project']['version']}")
