# Dataset : images + labels depuis CSV (colonnes pth, label). Les labels viennent du CSV uniquement.
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import yaml


class AffectNetDataset(Dataset):
    """Charge les images listées dans le CSV ; chaque ligne = pth (chemin relatif) + label (émotion)."""

    def __init__(self, root_dir, transform=None, labels_csv=None):
        """root_dir = base des données (ex. data/Train), labels_csv = chemin vers le CSV."""
        self.root_dir = root_dir
        self.transform = transform
        
        config_path = os.path.join("configs", "config.yaml")
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        self.classes = self.cfg['data']['class_names']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_paths = []
        self.multi_labels = []
        self.labels_df = None
        if labels_csv and os.path.exists(labels_csv):
            try:
                self.labels_df = pd.read_csv(labels_csv, index_col=0)
                print(f"[INFO] Labels CSV charge: {len(self.labels_df)} entrees")
            except Exception as e:
                print(f"[ERROR] Impossible de charger le CSV: {e}")
                raise ValueError("Le CSV des labels est obligatoire!")
        
        self._load_dataset()

    def _load_dataset(self):
        """Remplit image_paths et multi_labels à partir du CSV (chemins existants uniquement)."""
        if self.labels_df is None:
            print("[ERROR] Le CSV des labels est obligatoire pour ce dataset!")
            return
        
        for idx, row in self.labels_df.iterrows():
            try:
                relative_path = row['pth']
                csv_label = row['label']
                img_path = os.path.join(self.root_dir, relative_path)
                if not os.path.exists(img_path):
                    continue
                multi_hot = np.zeros(len(self.classes), dtype=np.float32)
                if csv_label in self.class_to_idx:
                    csv_idx = self.class_to_idx[csv_label]
                    multi_hot[csv_idx] = 1.0
                    
                    self.image_paths.append(img_path)
                    self.multi_labels.append(torch.tensor(multi_hot, dtype=torch.float32))
                
            except Exception:
                pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Retourne (image tensor, one-hot label). Image lue en grayscale, transform appliquée."""
        img_path = self.image_paths[idx]
        multi_label = self.multi_labels[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            h = self.cfg['data']['img_height']
            w = self.cfg['data']['img_width']
            img = np.zeros((h, w), dtype=np.uint8)

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, multi_label

if __name__ == "__main__":
    from torchvision import transforms
    print("[TEST] Verification AffectNetDataset...")
    dummy_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    try:
        with open("configs/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
            train_dir = cfg['data']['train_dir']

        ds = AffectNetDataset(
            root_dir=train_dir, 
            transform=dummy_transform,
            labels_csv="data/labels.csv"
        )
        print(f"[SUCCESS] Dataset charge. Nombre d'images : {len(ds)}")
        
        if len(ds) > 0:
            img, label = ds[0]
            emotion = ds.classes[int(torch.argmax(label).item())]
            print(f"   Shape image : {img.shape}")
            print(f"   Vecteur one-hot : {label}")
            print(f"   Emotion (label) : {emotion}")
        else:
            print("[WARNING] Le dataset est vide (verifiez data/labels.csv et les chemins).")
            
    except Exception as e:
        print(f"[ERROR] Echec du test : {e}")