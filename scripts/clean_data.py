# Déplace uniquement les images qui ont le bon label dans le CSV mais sont dans le mauvais dossier.
# Lit le CSV (pth, label), pour chaque image : si le dossier dans pth != label, déplace l'image
# dans le dossier correspondant au label et met à jour le chemin dans le CSV de sortie.

import pandas as pd
import os
import shutil

BASE_DIR = "./data"
CSV_INPUT = os.path.join(BASE_DIR, "labels.csv")
CSV_OUTPUT = os.path.join(BASE_DIR, "labels_cleaned.csv")


def normalize_split(pth):
    """Retourne (split, dossier_emotion, filename). Gère Train/train et Test/test.
    Si pth a 3 parties (ex. Train/anger/img.jpg) -> (Train, anger, img.jpg).
    Si pth a 2 parties (ex. anger/img.jpg) on ne peut pas savoir Train ou Test -> (None, None, None).
    """
    parts = pth.replace("\\", "/").strip("/").split("/")
    if len(parts) >= 3:
        split = parts[0]
        if split.lower() == "train":
            split = "Train"
        elif split.lower() == "test":
            split = "Test"
        return split, parts[1], parts[2]
    if len(parts) == 2:
        for split in ("Train", "Test"):
            full = os.path.join(BASE_DIR, split, parts[0], parts[1])
            if os.path.isfile(full):
                return split, parts[0], parts[1]
    return None, None, None


def move_to_correct_folder(current_rel_path, correct_label, base_dir):
    """
    Si l'image est dans un dossier qui ne correspond pas au label du CSV, la déplace
    dans le bon dossier (split/correct_label/filename). Retourne le nouveau chemin relatif.
    Ne déplace que si dossier actuel != label (les labels dans le CSV sont considérés corrects).
    """
    split, current_emotion, filename = normalize_split(current_rel_path)
    if split is None:
        return current_rel_path
    if current_emotion == correct_label:
        return current_rel_path

    old_full = os.path.join(base_dir, split, current_emotion, filename)
    new_folder = os.path.join(base_dir, split, correct_label)
    new_full = os.path.join(new_folder, filename)

    if not os.path.isfile(old_full):
        return current_rel_path

    os.makedirs(new_folder, exist_ok=True)
    shutil.move(old_full, new_full)
    return f"{split}/{correct_label}/{filename}"


def main():
    if not os.path.isfile(CSV_INPUT):
        print(f"[ERROR] Fichier non trouvé : {CSV_INPUT}")
        return

    df = pd.read_csv(CSV_INPUT, index_col=0)
    if "pth" not in df.columns or "label" not in df.columns:
        print("[ERROR] Le CSV doit contenir les colonnes 'pth' et 'label'.")
        return

    new_paths = []
    moved_count = 0

    for index, row in df.iterrows():
        pth = row["pth"]
        label = row["label"]
        new_pth = move_to_correct_folder(pth, label, BASE_DIR)
        if new_pth != pth:
            moved_count += 1
        new_paths.append(new_pth)
        if (index + 1) % 5000 == 0:
            print(f"Progression : {index + 1}/{len(df)}")

    df["pth"] = new_paths
    df.to_csv(CSV_OUTPUT, index=True)
    print(f"Nettoyage terminé. {moved_count} image(s) déplacée(s). Fichier enregistré : {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
