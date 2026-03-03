# HPO Optuna (lr, dropout, freeze_layers, hidden_size, etc.)
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("[ERROR] pip install optuna")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import AffectNetDataset
from src.model import get_model, set_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_config():
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg):
    with open(ROOT / "configs" / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def get_transforms(cfg, augmentation_strength=1.0):
    """Transforms train (avec augmentation) et val. augmentation_strength module rotation, jitter, erasing."""
    h, w = cfg["data"]["img_height"], cfg["data"]["img_width"]
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    
    rotation = int(15 * augmentation_strength)
    jitter = 0.2 * augmentation_strength
    erasing = 0.15 * augmentation_strength
    
    train_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(rotation),
        transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=jitter, contrast=jitter),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std]),
        transforms.RandomErasing(p=erasing),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std]),
    ])
    return train_tf, val_tf


def get_subset_loaders(cfg, sample_ratio=0.15, batch_size=64, augmentation_strength=1.0):
    """Charge un sous-échantillon du dataset (sample_ratio) pour accélérer les trials Optuna."""
    train_tf, val_tf = get_transforms(cfg, augmentation_strength)
    labels_csv = cfg["data"].get("labels_csv", "data/labels.csv")
    train_ds = AffectNetDataset(cfg["data"]["train_dir"], train_tf, labels_csv)
    val_ds = AffectNetDataset(cfg["data"]["test_dir"], val_tf, labels_csv)
    n_train = max(100, int(len(train_ds) * sample_ratio))
    n_val = max(50, int(len(val_ds) * sample_ratio))
    
    train_idx = np.random.choice(len(train_ds), min(n_train, len(train_ds)), replace=False)
    val_idx = np.random.choice(len(val_ds), min(n_val, len(val_ds)), replace=False)
    
    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader


class LabelSmoothingBCE(nn.Module):
    """BCEWithLogitsLoss avec lissage des labels (smoothing)."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + self.smoothing / 2
        return nn.functional.binary_cross_entropy_with_logits(pred, target)


def objective(trial, cfg, device, sample_ratio, full_search=False):
    """Fonction objectif Optuna : entraîne un modèle avec les hyperparamètres suggérés, retourne l'accuracy val."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    freeze_layers = trial.suggest_int("freeze_layers", 0, 4)
    hidden_size = trial.suggest_categorical("hidden_size", [0, 128, 256, 512])
    if full_search:
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        augmentation_strength = trial.suggest_float("augmentation_strength", 0.5, 1.5)
        pretrained = trial.suggest_categorical("pretrained", [True, False])
    else:
        batch_size = 32
        label_smoothing = 0.1
        augmentation_strength = 1.0
        pretrained = True
    
    set_seed(42)
    model = get_model(
        name="resnet18",
        num_classes=cfg["data"]["num_classes"],
        in_channels=cfg["data"]["channels"],
        dropout=dropout,
        pretrained=pretrained,
        freeze_layers=freeze_layers,
        hidden_size=hidden_size
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    train_loader, val_loader = get_subset_loaders(
        cfg, 
        sample_ratio=sample_ratio, 
        batch_size=batch_size,
        augmentation_strength=augmentation_strength
    )
    criterion = LabelSmoothingBCE(label_smoothing) if label_smoothing > 0 else nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_epochs = 8 if full_search else 5
    warmup_epochs = 1
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_acc = 0.0
    
    for epoch in range(n_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                correct += ((torch.sigmoid(out) > 0.5) == (y > 0.5)).sum().item()
                total += y.numel()
        
        acc = correct / total
        best_acc = max(best_acc, acc)
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Recherche d'hyperparametres Optuna")
    parser.add_argument("--trials", type=int, default=30, help="Nombre d'essais")
    parser.add_argument("--sample", type=float, default=0.15, help="Fraction du dataset (0.1 = 10%%)")
    parser.add_argument("--full", action="store_true", help="Recherche complete (plus d'hyperparametres)")
    parser.add_argument("--db", type=str, default=None, help="Chemin de la base SQLite pour reprendre")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"RECHERCHE D'HYPERPARAMETRES OPTUNA {'[COMPLETE]' if args.full else '[RAPIDE]'}")
    print(f"{'='*70}")
    print(f"Trials: {args.trials} | Dataset sample: {args.sample*100:.0f}%")
    print(f"{'='*70}")
    
    print("\nHyperparametres optimises:")
    print("  - Learning rate (lr)")
    print("  - Dropout")
    print("  - Weight decay")
    print("  - Freeze layers (0-4 couches gelees)")
    print("  - Hidden size (0, 128, 256, 512)")
    if args.full:
        print("  - Batch size")
        print("  - Label smoothing")
        print("  - Augmentation strength")
        print("  - Pretrained (True/False)")
    print(f"{'='*70}\n")
    
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    study_dir = ROOT / "weights" / "optuna"
    study_dir.mkdir(parents=True, exist_ok=True)
    if args.db:
        db_path = args.db
    else:
        db_path = f"sqlite:///{study_dir / 'study.db'}"
    sampler = TPESampler(multivariate=True, seed=42)
    pruner = HyperbandPruner(min_resource=2, max_resource=8 if args.full else 5)
    
    study = optuna.create_study(
        study_name="emotiscan_hpo",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=db_path,
        load_if_exists=True,
    )
    print(f"\n[Optuna] Lancement de {args.trials} essais...\n")
    
    study.optimize(
        lambda trial: objective(trial, cfg, device, args.sample, args.full),
        n_trials=args.trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    best = study.best_params
    print(f"\n{'='*70}")
    print(f"MEILLEURS HYPERPARAMETRES")
    print(f"{'='*70}")
    print(f"Accuracy (sur {args.sample*100:.0f}% du dataset): {study.best_value:.4f}")
    print(f"{'='*70}")
    print(f"Learning rate:     {best['lr']:.6f}")
    print(f"Dropout:           {best['dropout']:.3f}")
    print(f"Weight decay:      {best['weight_decay']:.6f}")
    print(f"Freeze layers:     {best['freeze_layers']} (sur 4)")
    print(f"Hidden size:       {best['hidden_size']}")
    if args.full:
        print(f"Batch size:        {best.get('batch_size', 32)}")
        print(f"Label smoothing:   {best.get('label_smoothing', 0.1):.3f}")
        print(f"Augmentation:      {best.get('augmentation_strength', 1.0):.2f}")
        print(f"Pretrained:        {best.get('pretrained', True)}")
    print(f"{'='*70}")
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\n[Stats] {n_complete} complete, {n_pruned} pruned (sur {len(study.trials)} total)")
    print(f"\n[Top 5 trials]")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, t in enumerate(trials_sorted[:5]):
        if t.value:
            print(f"  {i+1}. Acc={t.value:.4f} | freeze={t.params.get('freeze_layers', '?')} | hidden={t.params.get('hidden_size', '?')}")
    cfg["train"]["learning_rate"] = best["lr"]
    cfg["model"]["dropout"] = best["dropout"]
    cfg["train"]["weight_decay"] = best["weight_decay"]
    cfg["model"]["freeze_layers"] = best["freeze_layers"]
    cfg["model"]["hidden_size"] = best["hidden_size"]
    
    if args.full:
        cfg["train"]["batch_size"] = best.get("batch_size", 32)
        cfg["train"]["label_smoothing"] = best.get("label_smoothing", 0.1)
        cfg["model"]["pretrained"] = best.get("pretrained", True)
    
    save_config(cfg)
    print(f"\n[OK] Config mise a jour dans configs/config.yaml")
    results = {
        "best_accuracy": study.best_value,
        "best_params": best,
        "n_trials": len(study.trials),
        "n_complete": n_complete,
        "n_pruned": n_pruned,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(study_dir / "best_trial.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(study_dir / "best_summary.txt", "w") as f:
        f.write(f"EmotiScan - Recherche d'hyperparametres\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Trials: {len(study.trials)}\n")
        f.write(f"Best accuracy: {study.best_value:.4f}\n\n")
        f.write(f"Best hyperparameters:\n")
        for k, v in best.items():
            f.write(f"  {k}: {v}\n")
    
    print(f"\n[Next] Lance l'entrainement complet:")
    print(f"  python train.py")


if __name__ == "__main__":
    main()
