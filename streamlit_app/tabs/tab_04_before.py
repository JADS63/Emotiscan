import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from ..common import load_report_metrics, REPORTS_DIR


def render(cfg: Dict[str, Any], class_names: List[str], report_dir_name: str):
    st.markdown("### Resultats AVANT optimisation")
    
    # Charger les metriques
    metrics = load_report_metrics(report_dir_name)
    
    if metrics is None:
        st.warning(f"Pas de metriques trouvees dans `reports/{report_dir_name}/`")
        st.info("Lancez `python scripts/validate_model.py` pour generer les rapports.")
        return
    
    # Afficher les metriques principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        acc = metrics.get("accuracy", 0)
        st.metric("Accuracy", f"{acc:.2%}")
    
    with col2:
        # Compat: anciens reports -> metrics["macro"]["f1"]
        f1 = (
            metrics.get("f1_macro")
            or (metrics.get("macro", {}) or {}).get("f1")
            or metrics.get("f1")
            or 0
        )
        st.metric("F1-Score (macro)", f"{f1:.2%}")
    
    with col3:
        # Compat: anciens reports -> metrics["test_loss"]
        loss = metrics.get("loss", metrics.get("test_loss", metrics.get("val_loss", 0))) or 0
        st.metric("Loss", f"{loss:.4f}")
    
    # Images de la matrice de confusion
    report_path = REPORTS_DIR / report_dir_name
    
    st.markdown("#### Matrice de confusion")
    col1, col2 = st.columns(2)
    
    cm_path = report_path / "confusion_matrix.png"
    cm_norm_path = report_path / "confusion_matrix_normalized.png"
    
    with col1:
        if cm_path.exists():
            st.image(str(cm_path), caption="Matrice de confusion", use_container_width=True)
        else:
            st.info("Image non disponible")
    
    with col2:
        if cm_norm_path.exists():
            st.image(str(cm_norm_path), caption="Matrice normalisee", use_container_width=True)
        else:
            st.info("Image non disponible")
    
    # Metriques par classe
    st.markdown("#### Metriques par classe")
    
    csv_path = report_path / "classification_report.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)
    elif "per_class" in metrics:
        df = pd.DataFrame(metrics["per_class"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Rapport de classification non disponible")
    
    # Graphiques par classe
    st.markdown("#### Graphiques par classe")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        f1_path = report_path / "per_class_f1.png"
        if f1_path.exists():
            st.image(str(f1_path), caption="F1 par classe", use_container_width=True)
    
    with col2:
        prec_path = report_path / "per_class_precision.png"
        if prec_path.exists():
            st.image(str(prec_path), caption="Precision par classe", use_container_width=True)
    
    with col3:
        rec_path = report_path / "per_class_recall.png"
        if rec_path.exists():
            st.image(str(rec_path), caption="Recall par classe", use_container_width=True)
