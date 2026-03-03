import streamlit as st
from pathlib import Path
from typing import List
from collections import Counter


def render(class_names: List[str]):
    st.markdown("### Statistiques du Dataset")
    
    # Chemins des donnees
    ROOT = Path(__file__).parents[2]
    train_dir = ROOT / "data" / "Train"
    test_dir = ROOT / "data" / "Test"
    
    if not train_dir.exists():
        st.warning(f"Dossier de donnees introuvable: `{train_dir}`")
        st.info("""
        Pour utiliser cette fonctionnalite, placez vos donnees dans:
        - `data/Train/` (images d'entrainement)
        - `data/Test/` (images de test)
        
        Structure attendue:
        ```
        data/
          Train/
            anger/
            disgust/
            fear/
            happy/
            neutral/
            sad/
            surprise/
          Test/
            (meme structure)
        ```
        """)
        return
    
    # Compter les images par classe
    def count_images(base_dir: Path) -> dict:
        counts = {}
        for class_name in class_names:
            class_dir = base_dir / class_name
            if class_dir.exists():
                n = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                counts[class_name] = n
            else:
                counts[class_name] = 0
        return counts
    
    train_counts = count_images(train_dir)
    test_counts = count_images(test_dir)
    
    # Afficher les statistiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Entrainement")
        total_train = sum(train_counts.values())
        st.metric("Total images", f"{total_train:,}")
        
        for name, count in train_counts.items():
            pct = count / total_train * 100 if total_train > 0 else 0
            st.write(f"**{name}**: {count:,} ({pct:.1f}%)")
    
    with col2:
        st.markdown("#### Test")
        total_test = sum(test_counts.values())
        st.metric("Total images", f"{total_test:,}")
        
        for name, count in test_counts.items():
            pct = count / total_test * 100 if total_test > 0 else 0
            st.write(f"**{name}**: {count:,} ({pct:.1f}%)")
    
    # Graphique de distribution
    st.markdown("#### Distribution des classes")
    
    import pandas as pd
    
    df = pd.DataFrame({
        "Classe": class_names,
        "Train": [train_counts[c] for c in class_names],
        "Test": [test_counts[c] for c in class_names],
    })
    
    st.bar_chart(df.set_index("Classe"))
    
    # Ratio train/test
    st.markdown("#### Ratio Train/Test")
    total = total_train + total_test
    if total > 0:
        train_ratio = total_train / total * 100
        test_ratio = total_test / total * 100
        st.write(f"- **Train**: {train_ratio:.1f}% ({total_train:,} images)")
        st.write(f"- **Test**: {test_ratio:.1f}% ({total_test:,} images)")
