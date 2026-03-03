# Stubs audio (onglet vidéo)
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def extract_audio_mono_16k(path: Path) -> Tuple[np.ndarray, int]:
    return np.array([], dtype=np.float32), 16000


def segment_audio(audio: np.ndarray, sr: int, window_sec: float, hop_sec: float) -> List[np.ndarray]:
    return []

def load_audio_emotion_model(model_id: str, device):
    return None, None, None

def predict_audio_probs(segment: np.ndarray, sr: int, model, processor) -> np.ndarray:
    return np.array([])

def audio_window_index(t: float, sr: int, hop_sec: float) -> int:
    return 0

def compute_vad_flags(segments: List[np.ndarray], sr: int) -> List[bool]:
    return [False] * len(segments)
