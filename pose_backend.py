"""pose_backend.py - MediaPipe PoseLandmarker 래퍼 (모델 파일 관리, 생성, 그리기 도우미).

MediaPipe 1.0부터는 예전 mp.solutions.pose API가 없어졌으므로 Tasks API를 쓴다.
모델 파일(.task)은 models/ 폴더에 두며, 없으면 자동으로 내려받는다.
"""
from __future__ import annotations

import os

os.environ.setdefault("GLOG_minloglevel", "2")      # MediaPipe C++ 경고 로그 줄이기
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from jump_counter import Landmark

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_NAMES = ("lite", "full", "heavy")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{m}/float16/latest/pose_landmarker_{m}.task")

# 얼굴 연결선을 뺀 몸통·팔·다리 연결선 (랜드마크 11번부터가 몸)
BODY_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS
                    if c.start >= 11 and c.end >= 11]


def ensure_model(name: str) -> Path:
    """models/pose_landmarker_<name>.task 경로를 돌려준다. 없으면 내려받는다."""
    if name not in MODEL_NAMES:
        raise ValueError(f"모델 이름은 {MODEL_NAMES} 중 하나여야 합니다: {name}")
    path = MODEL_DIR / f"pose_landmarker_{name}.task"
    if not path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        url = MODEL_URL.format(m=name)
        print(f"[모델] {path.name} 내려받는 중... {url}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def create_landmarker(model: str = "full", live: bool = False, result_callback=None,
                      min_confidence: float = 0.5):
    """PoseLandmarker 생성. live=True 면 LIVE_STREAM(콜백) 모드, 아니면 VIDEO 모드."""
    mode = vision.RunningMode.LIVE_STREAM if live else vision.RunningMode.VIDEO
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model(model))),
        running_mode=mode,
        num_poses=1,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
        result_callback=result_callback,
    )
    return vision.PoseLandmarker.create_from_options(options)


def to_mp_image(bgr: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def detect_landmarks(landmarker, bgr: np.ndarray, timestamp_ms: int):
    """VIDEO 모드 추론. 첫 번째 사람의 랜드마크 목록(33개) 또는 None."""
    result = landmarker.detect_for_video(to_mp_image(bgr), timestamp_ms)
    return result.pose_landmarks[0] if result.pose_landmarks else None


def landmarks_to_tuples(landmarks) -> list[Landmark]:
    """MediaPipe 랜드마크 객체를 스레드 간에 넘기기 쉬운 가벼운 튜플로 바꾼다 (정규화 좌표)."""
    out = []
    for lm in landmarks:
        v = lm.visibility
        out.append(Landmark(lm.x, lm.y, 1.0 if v is None else float(v)))
    return out


def draw_skeleton(img: np.ndarray, landmarks, min_visibility: float = 0.5,
                  color=(0, 200, 255)) -> None:
    """정규화 랜드마크를 img 크기에 맞춰 몸 스켈레톤을 그린다."""
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h), lm.visibility) for lm in landmarks]
    thick = max(1, int(h / 360))
    for a, b in BODY_CONNECTIONS:
        if pts[a][2] >= min_visibility and pts[b][2] >= min_visibility:
            cv2.line(img, pts[a][:2], pts[b][:2], color, thick, cv2.LINE_AA)
    r = max(2, int(h / 240))
    for p in pts[11:]:
        if p[2] >= min_visibility:
            cv2.circle(img, p[:2], r, (255, 255, 255), -1, cv2.LINE_AA)
