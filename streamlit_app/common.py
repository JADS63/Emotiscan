# Config, chargement modèle, preprocessing, inférence (partagé par les onglets Streamlit)
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union

import yaml
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import get_model, EmotiScanResNet18

try:
    import cv2
except ImportError:
    cv2 = None

WEIGHTS_BEFORE_DEFAULT = ROOT / "weights" / "best_model_without_optimisation.pth"
WEIGHTS_AFTER_DEFAULT = ROOT / "weights" / "best_model_with_optimisation.pth"
REPORTS_DIR = ROOT / "reports"


def load_config() -> Dict[str, Any]:
    """Charge configs/config.yaml."""
    config_path = ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_report_dirs() -> List[str]:
    """Liste les dossiers dans reports/ qui contiennent metrics.json (triés par date, plus récent en premier)."""
    if not REPORTS_DIR.exists():
        return []
    report_dirs: List[Path] = [d for d in REPORTS_DIR.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    report_dirs.sort(key=lambda d: (d / "metrics.json").stat().st_mtime, reverse=True)
    return [d.name for d in report_dirs]


def load_report_metrics(report_dir_name: str) -> Optional[Dict[str, Any]]:
    """Charge les métriques d'un rapport (reports/<report_dir_name>/metrics.json)."""
    metrics_path = REPORTS_DIR / report_dir_name / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_state_dict_any(checkpoint) -> dict:
    """Extrait le state_dict quel que soit le format (state_dict, model_state_dict, model, ou direct)."""
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            return checkpoint["model"]
        return checkpoint
    return checkpoint


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_path(cfg: Dict[str, Any], weights_path: Path) -> Tuple[Optional[EmotiScanResNet18], Optional[torch.device], Optional[dict], Optional[Path]]:
    """Charge le modèle depuis un fichier .pth. Retourne (model, device, checkpoint, path) ou (None, None, None, None) si erreur."""
    device = get_device()
    if not weights_path.is_absolute():
        weights_path = Path(ROOT) / weights_path
    
    if not weights_path.exists():
        return None, None, None, None
    model = get_model(
        name=cfg["model"].get("name", "resnet18"),
        num_classes=cfg["data"]["num_classes"],
        in_channels=cfg["data"]["channels"],
        dropout=cfg["model"].get("dropout", 0.3),
        pretrained=False,
        freeze_layers=cfg["model"].get("freeze_layers", 0),
        hidden_size=cfg["model"].get("hidden_size", 0)
    ).to(device)
    
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = _load_state_dict_any(checkpoint)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"[WARN] Chargement non-strict: {e}")
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, device, checkpoint, weights_path
    except Exception:
        return None, None, None, None


def get_inference_transform(cfg: Dict[str, Any]) -> transforms.Compose:
    """Resize 96x96 + ToTensor + Normalize (mean, std de la config)."""
    h, w = cfg["data"]["img_height"], cfg["data"]["img_width"]
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std]),
    ])


def preprocess_image(image: Image.Image, cfg: Dict[str, Any]) -> torch.Tensor:
    """Convertit une image PIL en tensor (1, 1, H, W) pour l'inférence."""
    if image.mode != "L":
        image = image.convert("L")
    transform = get_inference_transform(cfg)
    tensor = transform(image)
    return tensor.unsqueeze(0)


def to_grayscale_np(img_rgb: np.ndarray) -> np.ndarray:
    """Convertit une image RGB (H, W, 3) en niveaux de gris uint8."""
    if cv2 is None:
        return np.array(Image.fromarray(img_rgb).convert("L"), dtype=np.uint8)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def auto_face_crop_rgb(img_rgb: np.ndarray) -> Tuple[np.ndarray, Optional[tuple]]:
    """Détection visage (Haar cascade). Retourne (crop RGB avec marge 20%, bbox) ou (image entière, None)."""
    if cv2 is None:
        return img_rgb, None
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return img_rgb, None
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        for scale_factor, min_neighbors, min_size in [
            (1.05, 3, (20, 20)),
            (1.1, 4, (30, 30)),
            (1.2, 5, (40, 40)),
        ]:
            faces = cascade.detectMultiScale(
                gray_eq, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
            )
            if len(faces) == 0:
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
                )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                margin = int(0.2 * max(w, h))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(img_rgb.shape[1], x + w + margin)
                y2 = min(img_rgb.shape[0], y + h + margin)
                crop = img_rgb[y1:y2, x1:x2]
                return crop, (x1, y1, x2 - x1, y2 - y1)
        return img_rgb, None
    except Exception:
        return img_rgb, None


def predict_emotion(model, x_or_image, device, class_names, cfg=None) -> Tuple[str, float, np.ndarray]:
    """Retourne (nom_classe, confiance, tableau des probas). x_or_image = tensor ou PIL Image."""
    model.eval()
    if cfg is not None and isinstance(x_or_image, Image.Image):
        x = preprocess_image(x_or_image, cfg).to(device)
    else:
        x = x_or_image.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs


def predict_batch(
    model: EmotiScanResNet18,
    images: List[Image.Image],
    cfg: Dict[str, Any],
    device: torch.device,
    class_names: List[str],
) -> List[Tuple[str, np.ndarray]]:
    """Prédit les émotions pour une liste d'images. Retourne [(nom_classe, probs), ...]."""
    results = []
    for img in images:
        x = preprocess_image(img, cfg)
        pred_name, _, probs = predict_emotion(model, x, device, class_names)
        results.append((pred_name, probs))
    return results


def format_delta(before: float, after: float) -> str:
    """Formate la différence (after - before) en HTML avec classe delta-pos (vert) ou delta-neg (rouge)."""
    delta = after - before
    sign = "+" if delta >= 0 else ""
    css_class = "delta-pos" if delta >= 0 else "delta-neg"
    return f'<span class="{css_class}">{sign}{delta:.2%}</span>'


def get_emotion_emoji(emotion: str) -> str:
    """Retourne l'emoji associé à une émotion (anger -> 😠, etc.)."""
    emojis = {
        "anger": "😠",
        "disgust": "🤢", 
        "fear": "😨",
        "happy": "😊",
        "neutral": "😐",
        "sad": "😢",
        "surprise": "😲",
    }
    return emojis.get(emotion.lower(), "🙂")
