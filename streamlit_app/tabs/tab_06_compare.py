import streamlit as st
from typing import Dict, Any, List

from ..common import load_report_metrics, format_delta


def _get_f1_macro(metrics: Dict[str, Any]) -> float:
    return (
        metrics.get("f1_macro")
        or (metrics.get("macro", {}) or {}).get("f1")
        or metrics.get("f1")
        or 0.0
    )


def _get_loss(metrics: Dict[str, Any]) -> float:
    return metrics.get("loss", metrics.get("test_loss", metrics.get("val_loss", 0.0))) or 0.0


def render(
    cfg: Dict[str, Any], 
    class_names: List[str], 
    before_report_dir: str, 
    after_report_dir: str
):
    st.markdown("### Comparaison AVANT vs APRES optimisation")
    
    # Charger les metriques
    metrics_before = load_report_metrics(before_report_dir)
    metrics_after = load_report_metrics(after_report_dir)
    
    if metrics_before is None or metrics_after is None:
        st.warning("Metriques manquantes. Selectionnez des rapports valides dans la sidebar.")
        return
    
    # Metriques principales avec delta
    st.markdown("#### Metriques principales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        acc_before = metrics_before.get("accuracy", 0)
        acc_after = metrics_after.get("accuracy", 0)
        delta = acc_after - acc_before
        st.metric(
            "Accuracy", 
            f"{acc_after:.2%}", 
            delta=f"{delta:+.2%}",
            delta_color="normal"
        )
    
    with col2:
        f1_before = _get_f1_macro(metrics_before)
        f1_after = _get_f1_macro(metrics_after)
        delta = f1_after - f1_before
        st.metric(
            "F1-Score (macro)", 
            f"{f1_after:.2%}", 
            delta=f"{delta:+.2%}",
            delta_color="normal"
        )
    
    with col3:
        loss_before = _get_loss(metrics_before)
        loss_after = _get_loss(metrics_after)
        delta = loss_after - loss_before
        st.metric(
            "Loss", 
            f"{loss_after:.4f}", 
            delta=f"{delta:+.4f}",
            delta_color="inverse"  # Lower is better
        )
    
    # Tableau comparatif
    st.markdown("#### Resume")
    
    comparison_data = {
        "Metrique": ["Accuracy", "F1-Score", "Loss"],
        "AVANT": [
            f"{acc_before:.2%}",
            f"{f1_before:.2%}",
            f"{loss_before:.4f}"
        ],
        "APRES": [
            f"{acc_after:.2%}",
            f"{f1_after:.2%}",
            f"{loss_after:.4f}"
        ],
        "Delta": [
            f"{acc_after - acc_before:+.2%}",
            f"{f1_after - f1_before:+.2%}",
            f"{loss_after - loss_before:+.4f}"
        ]
    }
    
    st.table(comparison_data)
    
    # Conclusion
    st.markdown("#### Conclusion")
    
    improvement = acc_after - acc_before
    if improvement > 0.01:  # > 1%
        st.success(f"L'optimisation a ameliore l'accuracy de **{improvement:.2%}**.")
    elif improvement > 0:
        st.info(f"Legere amelioration de **{improvement:.2%}**.")
    elif improvement == 0:
        st.warning("Pas d'amelioration significative.")
    else:
        st.error(f"Regression de **{abs(improvement):.2%}**. Verifiez les hyperparametres.")
