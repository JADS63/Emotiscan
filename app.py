# App Streamlit : démo image/vidéo/webcam, résultats, config
import streamlit as st

from streamlit_app.common import (
    load_config,
    WEIGHTS_BEFORE_DEFAULT,
    WEIGHTS_AFTER_DEFAULT,
    list_report_dirs,
)

from streamlit_app.tabs import (
    tab_00_presentation,
    tab_01_image,
    tab_02_video,
    tab_04_before,
    tab_05_after,
    tab_06_compare,
    tab_07_config,
    tab_08_dataset,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STREAMLIT CONFIG + STYLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(page_title="EmotiScan", page_icon="🙂", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.section-title { font-size: 1.10rem; font-weight: 700; margin: 0.25rem 0 0.75rem 0; }
.card { background: #f6f7f9; border: 1px solid #e7e8ec; padding: 1rem; border-radius: 10px; }
.small { color: #5f6368; font-size: 0.9rem; }
hr { margin: 1.25rem 0; }
.kpi { font-size: 1.4rem; font-weight: 800; }
.delta-pos { color: #0f9d58; font-weight: 700; }
.delta-neg { color: #d93025; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER + CONFIG LOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.title("EmotiScan — Démo & Analyse")
st.write("Interface de test, d’analyse, d’évaluation et de comparaison AVANT/APRÈS optimisation.")

cfg = load_config()
class_names = cfg["data"]["class_names"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR (paramètres globaux)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.header("Paramètres")

st.sidebar.caption("Les onglets Résultats/Comparaison lisent les fichiers déjà calculés dans `reports/`.")

st.sidebar.subheader("Checkpoints")
before_path = st.sidebar.text_input("Modèle AVANT optimisation", value=str(WEIGHTS_BEFORE_DEFAULT))
after_path = st.sidebar.text_input("Modèle APRÈS optimisation", value=str(WEIGHTS_AFTER_DEFAULT))

st.sidebar.subheader("Reports (performances déjà calculées)")
report_dirs = list_report_dirs()
if len(report_dirs) == 0:
    st.sidebar.warning("Aucun dossier dans reports/. Lance `python scripts/04_validate_model.py` pour en générer.")
    before_report = ""
    after_report = ""
else:
    # defaults: AVANT = dossier sans optimisation si présent, sinon le + récent
    default_before = "validation_model_without_optimisation" if "validation_model_without_optimisation" in report_dirs else report_dirs[0]
    default_after = report_dirs[0]

    before_report = st.sidebar.selectbox("Report AVANT", options=report_dirs, index=report_dirs.index(default_before))
    after_report = st.sidebar.selectbox("Report APRÈS", options=report_dirs, index=report_dirs.index(default_after))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NAVIGATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tabs = st.tabs(
    [
        "0) Présentation",
        "1) Démo Image",
        "2) Démo Vidéo",
        "3) Résultats AVANT",
        "4) Résultats APRÈS",
        "5) Comparaison AVANT vs APRÈS",
        "6) Dataset (stats)",
        "7) Config",
    ]
)

with tabs[0]:
    tab_00_presentation.render(cfg)

with tabs[1]:
    tab_01_image.render(cfg, class_names, after_path=after_path)

with tabs[2]:
    tab_02_video.render(cfg, class_names, after_path=after_path)

with tabs[3]:
    if before_report:
        tab_04_before.render(cfg, class_names, report_dir_name=before_report)
    else:
        st.info("Aucun report sélectionné.")

with tabs[4]:
    if after_report:
        tab_05_after.render(cfg, class_names, report_dir_name=after_report)
    else:
        st.info("Aucun report sélectionné.")

with tabs[5]:
    if before_report and after_report:
        tab_06_compare.render(cfg, class_names, before_report_dir=before_report, after_report_dir=after_report)
    else:
        st.info("Sélectionne deux reports (AVANT + APRÈS) dans la sidebar.")

with tabs[6]:
    tab_08_dataset.render(class_names)

with tabs[7]:
    tab_07_config.render(cfg, before_path=before_path, after_path=after_path)


# FOOTER
st.markdown("---")
st.markdown(
    "<div class='small'>EmotiScan — Streamlit + PyTorch. Démo, analyse, comparaison AVANT/APRÈS optimisation.</div>",
    unsafe_allow_html=True,
)
