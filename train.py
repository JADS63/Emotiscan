# Entraînement du modèle. Option --tune pour recherche hyperparamètres (Optuna).
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset import AffectNetDataset
from src.model import get_model, set_seed, save_checkpoint

ROOT = Path(__file__).resolve().parent


def load_config() -> Dict[str, Any]:
    """Charge configs/config.yaml."""
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict[str, Any]):
    """Écrit la config dans configs/config.yaml."""
    with open(ROOT / "configs" / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def get_transforms(cfg: Dict[str, Any], train: bool = True):
    """Transforms : resize 96x96, normalisation. En train : flip, rotation, color jitter, random erasing."""
    h, w = cfg["data"]["img_height"], cfg["data"]["img_width"]
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    
    if train:
        aug = cfg.get("data", {})
        return transforms.Compose([
            transforms.Resize((h, w)),
            transforms.RandomHorizontalFlip(p=aug.get("hflip_p", 0.5)),
            transforms.RandomRotation(aug.get("aug_rotation_deg", 15)),
            transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=aug.get("aug_color_jitter", 0.2), contrast=aug.get("aug_color_jitter", 0.2)),
            transforms.ToTensor(),
            transforms.Normalize([mean], [std]),
            transforms.RandomErasing(p=aug.get("aug_random_erasing", 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std]),
    ])


def get_loaders(cfg: Dict[str, Any]):
    """DataLoaders train et val à partir du CSV (data/labels.csv par défaut)."""
    bs = cfg["train"]["batch_size"]
    labels_csv = cfg["data"].get("labels_csv", "data/labels.csv")
    train_ds = AffectNetDataset(cfg["data"]["train_dir"], get_transforms(cfg, True), labels_csv)
    val_ds = AffectNetDataset(cfg["data"]["test_dir"], get_transforms(cfg, False), labels_csv)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True),
    )


def train_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    """Une epoch d'entraînement. Retourne (loss_moyenne, accuracy)."""
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += ((torch.sigmoid(out) > 0.5) == (y > 0.5)).sum().item()
        total += y.numel()
    return loss_sum / total, correct / total


def validate(model, loader, criterion, device) -> Tuple[float, float]:
    """Validation. Retourne (loss, accuracy)."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            correct += ((torch.sigmoid(out) > 0.5) == (y > 0.5)).sum().item()
            total += y.numel()
    return loss_sum / total, correct / total


def train(cfg, lr=None, dropout=None, weight_decay=None, epochs=None, verbose=True, resume=True) -> float:
    """Boucle d'entraînement avec early stopping. Sauvegarde best_model.pth et last_checkpoint.pth."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["project"]["seed"])
    
    lr = lr or cfg["train"]["learning_rate"]
    dropout = dropout or cfg["model"]["dropout"]
    weight_decay = weight_decay or cfg["train"]["weight_decay"]
    epochs = epochs or cfg["train"]["epochs"]
    freeze_layers = cfg["model"].get("freeze_layers", 0)
    hidden_size = cfg["model"].get("hidden_size", 0)
    
    model = get_model(
        name="resnet18",  # Toujours ResNet18
        num_classes=cfg["data"]["num_classes"], 
        in_channels=cfg["data"]["channels"], 
        dropout=dropout, 
        pretrained=cfg["model"].get("pretrained", True),
        freeze_layers=freeze_layers,
        hidden_size=hidden_size
    ).to(device)
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    
    if verbose:
        print(f"[Model] ResNet18 | {n_total/1e6:.1f}M params ({n_trainable/1e6:.1f}M trainable) | {device}")
        print(f"[Config] freeze_layers={freeze_layers} | hidden_size={hidden_size} | dropout={dropout:.3f}")
    
    train_loader, val_loader = get_loaders(cfg)
    if verbose:
        print(f"[Data] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler: Cosine Annealing
    def lr_lambda(epoch):
        return 0.5 * (1 + np.cos(np.pi * epoch / epochs))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_acc, patience, start_epoch = 0.0, 0, 0
    (ROOT / "weights").mkdir(exist_ok=True)
    
    # Reprise automatique depuis le dernier checkpoint
    last_ckpt = ROOT / "weights" / "last_checkpoint.pth"
    if resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_acc = ckpt.get("best_acc", 0.0)
        patience = ckpt.get("patience", 0)
        if verbose:
            print(f"[Resume] Reprise depuis epoch {start_epoch} (best_acc: {best_acc:.4f})")
    
    for epoch in range(start_epoch, epochs):
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        if verbose:
            print(f"Epoch {epoch+1:02d}/{epochs} | LR: {current_lr:.6f} | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f}")
        
        # Sauvegarde du meilleur modèle
        state = {"epoch": epoch+1, "state_dict": model.state_dict(), "best_acc": val_acc}
        if val_acc > best_acc:
            best_acc, patience = val_acc, 0
            save_checkpoint(state, "weights/best_model.pth")
            if verbose:
                print(f"  -> Best! ({best_acc:.4f})")
        else:
            patience += 1
        
        save_checkpoint({
            "epoch": epoch+1, 
            "state_dict": model.state_dict(), 
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "patience": patience,
        }, "weights/last_checkpoint.pth")
        
        if patience >= cfg["train"]["early_stopping_patience"]:
            if verbose:
                print(f"Early stop @ epoch {epoch+1}")
            break
    
    return best_acc


def tune(cfg, n_trials=20):
    """Tuning Optuna rapide (lr, dropout, weight_decay, freeze_layers, hidden_size) puis entraînement."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[ERROR] pip install optuna")
        return {}
    
    print(f"\n{'='*50}\nTUNING OPTUNA RAPIDE - {n_trials} trials\n{'='*50}")
    print("[INFO] Pour une recherche complete (architecture, couches gelees, etc.):")
    print("       python scripts/tune_optuna.py --trials 50 --full\n")
    
    def objective(trial):
        test_cfg = cfg.copy()
        test_cfg["model"] = cfg["model"].copy()
        test_cfg["model"]["freeze_layers"] = trial.suggest_int("freeze_layers", 0, 3)
        test_cfg["model"]["hidden_size"] = trial.suggest_categorical("hidden_size", [0, 256])
        
        return train(
            test_cfg, 
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            dropout=trial.suggest_float("dropout", 0.2, 0.5),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            epochs=10, 
            verbose=False,
            resume=False
        )
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best = study.best_params
    print(f"\n[BEST] Acc: {study.best_value:.4f}")
    print(f"  lr={best['lr']:.6f} | dropout={best['dropout']:.3f}")
    print(f"  freeze_layers={best['freeze_layers']} | hidden_size={best['hidden_size']}")
    
    return best


def evaluate(cfg):
    """Évalue le modèle (weights/best_model.pth) sur le jeu de validation et affiche accuracy par classe."""
    print(f"\n{'='*50}\nEVALUATION\n{'='*50}\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, val_loader = get_loaders(cfg)
    class_names = cfg["data"]["class_names"]
    freeze_layers = cfg["model"].get("freeze_layers", 0)
    hidden_size = cfg["model"].get("hidden_size", 0)
    
    ckpt_path = ROOT / "weights" / "best_model.pth"
    if not ckpt_path.exists():
        print("[ERROR] No model found")
        return None
    
    model = get_model(
        name="resnet18",
        num_classes=cfg["data"]["num_classes"], 
        in_channels=cfg["data"]["channels"],
        dropout=cfg["model"]["dropout"], 
        pretrained=False,
        freeze_layers=freeze_layers,
        hidden_size=hidden_size
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Eval"):
            x = x.to(device)
            out = model(x)
            y_true.extend(torch.argmax(y, 1).tolist())
            y_pred.extend(torch.argmax(out, 1).cpu().tolist())
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean()
    
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    for i, name in enumerate(class_names):
        tp = ((y_true == i) & (y_pred == i)).sum()
        fp = ((y_true != i) & (y_pred == i)).sum()
        fn = ((y_true == i) & (y_pred != i)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*p*r / (p + r) if (p + r) > 0 else 0
        print(f"  {name:12s} | P:{p:.3f} R:{r:.3f} F1:{f1:.3f}")
    
    (ROOT / "reports" / "latest").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "reports" / "latest" / "metrics.json", "w") as f:
        json.dump({"accuracy": float(acc)}, f, indent=2)
    
    return acc


def main():
    """Point d'entrée : parse les arguments (--tune, --trials, --eval-only, --fresh) et lance train / tune / evaluate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Lancer optimisation Optuna")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--eval-only", action="store_true", help="Evaluation seule")
    parser.add_argument("--fresh", action="store_true", help="Repartir de zero (ignorer checkpoint)")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("EMOTISCAN TRAINING")
    print("="*60 + "\n")
    
    cfg = load_config()
    
    if args.eval_only:
        evaluate(cfg)
        return
    
    if args.tune:
        best = tune(cfg, args.trials)
        if best:
            cfg["train"]["learning_rate"] = best["lr"]
            cfg["model"]["dropout"] = best["dropout"]
            cfg["train"]["weight_decay"] = best["weight_decay"]
            cfg["model"]["freeze_layers"] = best.get("freeze_layers", 0)
            cfg["model"]["hidden_size"] = best.get("hidden_size", 0)
            save_config(cfg)
            print("[OK] Config mise a jour avec les meilleurs hyperparametres")
    
    # Supprimer checkpoint si --fresh
    if args.fresh:
        ckpt = ROOT / "weights" / "last_checkpoint.pth"
        if ckpt.exists():
            ckpt.unlink()
            print("[Fresh] Checkpoint supprimé, démarrage de zéro")
    
    print(f"\n{'='*60}\nTRAINING\n{'='*60}\n")
    best_acc = train(cfg, resume=not args.fresh)
    print(f"\n[DONE] Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    # Evaluation finale
    evaluate(cfg)


if __name__ == "__main__":
    main()
