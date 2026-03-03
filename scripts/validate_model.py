# Eval checkpoint -> metrics.json, confusion matrix, graphiques
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import AffectNetDataset
from src.model import get_model


def load_config():
    """Charge configs/config.yaml."""
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_val_transform(cfg):
    """Transforms de validation : resize, ToTensor, Normalize (sans augmentation)."""
    return transforms.Compose([
        transforms.Resize((cfg["data"]["img_height"], cfg["data"]["img_width"])),
        transforms.ToTensor(),
        transforms.Normalize([cfg["data"]["mean"]], [cfg["data"]["std"]]),
    ])


def compute_metrics(y_true, y_pred, class_names):
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    metrics = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        })
    
    return cm, metrics


def plot_confusion_matrix(cm, class_names, output_path, normalize=False, title="Confusion Matrix"):
    """Génère et enregistre la matrice de confusion en PNG (normale ou normalisée)."""
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_plot = np.nan_to_num(cm_plot)
    else:
        cm_plot = cm
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_plot, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='Vrai label',
           xlabel='Prédiction')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm_plot.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm_plot[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm_plot[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_per_class_metric(values, class_names, metric_name, output_path):
    """Graphique en barres : une métrique (precision, recall ou f1) par classe, sauvegardé en PNG."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(class_names))
    bars = ax.bar(x, values, color='steelblue')
    
    ax.set_xlabel('Classes')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} par classe')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """Charge le checkpoint, évalue sur le jeu de test, génère metrics.json, matrices de confusion et graphiques par classe."""
    parser = argparse.ArgumentParser(description="Validation du modèle")
    parser.add_argument("--checkpoint", type=str, default="weights/best_model.pth",
                       help="Chemin vers le checkpoint")
    parser.add_argument("--output", type=str, default=None,
                       help="Dossier de sortie pour les rapports")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("VALIDATION DU MODELE")
    print(f"{'='*60}\n")
    
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["data"]["class_names"]
    n_classes = len(class_names)
    ckpt_path = ROOT / args.checkpoint
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint non trouvé: {ckpt_path}")
        return
    
    print(f"[INFO] Checkpoint: {ckpt_path}")
    print(f"[INFO] Device: {device}")
    model = get_model(
        name="resnet18",
        num_classes=cfg["data"]["num_classes"],
        in_channels=cfg["data"]["channels"],
        dropout=cfg["model"]["dropout"],
        pretrained=False,
        freeze_layers=cfg["model"].get("freeze_layers", 0),
        hidden_size=cfg["model"].get("hidden_size", 0)
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Modèle chargé: ResNet18 ({n_params/1e6:.1f}M params)")
    labels_csv = cfg["data"].get("labels_csv", "data/labels.csv")
    test_ds = AffectNetDataset(cfg["data"]["test_dir"], get_val_transform(cfg), labels_csv)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], 
                            shuffle=False, num_workers=0)
    
    print(f"[INFO] Test dataset: {len(test_ds)} images\n")
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluation"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            
            pred = torch.argmax(out, dim=1).cpu()
            true = torch.argmax(y, dim=1).cpu()
            y_true.extend(true.tolist())
            y_pred.extend(pred.tolist())
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    accuracy = (y_true == y_pred).mean()
    test_loss = total_loss / len(test_ds)
    cm, per_class = compute_metrics(y_true, y_pred, class_names)
    precisions = [m["precision"] for m in per_class]
    recalls = [m["recall"] for m in per_class]
    f1s = [m["f1"] for m in per_class]
    supports = [m["support"] for m in per_class]
    
    macro = {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1s),
    }
    
    total_support = sum(supports)
    weighted = {
        "precision": sum(p * s for p, s in zip(precisions, supports)) / total_support,
        "recall": sum(r * s for r, s in zip(recalls, supports)) / total_support,
        "f1": sum(f * s for f, s in zip(f1s, supports)) / total_support,
    }
    print(f"\n{'='*60}")
    print("RESULTATS")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nMacro F1: {macro['f1']:.4f}")
    print(f"Weighted F1: {weighted['f1']:.4f}")
    print(f"\nPer-class metrics:")
    print("-" * 50)
    for m in per_class:
        print(f"  {m['class']:12s} | P:{m['precision']:.3f} R:{m['recall']:.3f} F1:{m['f1']:.3f} | n={m['support']}")
    if args.output:
        output_dir = ROOT / args.output
    else:
        output_dir = ROOT / "reports" / "latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Sauvegarde des rapports dans: {output_dir}")
    metrics_json = {
        "checkpoint": str(args.checkpoint),
        "n_test": len(test_ds),
        "test_loss": test_loss,
        "loss": test_loss,
        "accuracy": accuracy,
        "precision_macro": macro["precision"],
        "recall_macro": macro["recall"],
        "f1_macro": macro["f1"],
        "precision_weighted": weighted["precision"],
        "recall_weighted": weighted["recall"],
        "f1_weighted": weighted["f1"],
        "macro": macro,
        "weighted": weighted,
        "per_class": {
            "class_names": class_names,
            "precision": precisions,
            "recall": recalls,
            "f1": f1s,
            "support": supports,
        },
        "confusion_matrix": cm.tolist(),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    df_report = pd.DataFrame(per_class)
    df_report.to_csv(output_dir / "classification_report.csv", index=False)
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png", 
                         normalize=False, title="Matrice de confusion")
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix_normalized.png",
                         normalize=True, title="Matrice de confusion (normalisée)")
    plot_per_class_metric(precisions, class_names, "Precision", output_dir / "per_class_precision.png")
    plot_per_class_metric(recalls, class_names, "Recall", output_dir / "per_class_recall.png")
    plot_per_class_metric(f1s, class_names, "F1-Score", output_dir / "per_class_f1.png")
    
    print(f"\n[OK] Rapports générés:")
    print(f"  - metrics.json")
    print(f"  - classification_report.csv")
    print(f"  - confusion_matrix.png")
    print(f"  - confusion_matrix_normalized.png")
    print(f"  - per_class_precision.png")
    print(f"  - per_class_recall.png")
    print(f"  - per_class_f1.png")


if __name__ == "__main__":
    main()
