from __future__ import annotations
import logging
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QListWidget, QHBoxLayout, QSlider
from PySide6.QtCore import Qt, Slot
from core.cache_manager import CacheManager
import cv2
import numpy as np
import insightface

logger = logging.getLogger("ui.identity")

class IdentityPage(QWidget):
    def __init__(self, pipeline_manager):
        super().__init__()
        self.pipeline = pipeline_manager
        self.cache = CacheManager()
        self.model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640,640))
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Identity Page"))
        btn = QPushButton("Upload reference image")
        btn.clicked.connect(self.on_upload)
        layout.addWidget(btn)
        self.ref_list = QListWidget()
        layout.addWidget(self.ref_list)
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setValue(80)
        layout.addWidget(QLabel("Identity Strength"))
        layout.addWidget(self.strength_slider)

    @Slot()
    def on_upload(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select reference image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        self._process_reference(path)
        self.ref_list.addItem(path)

    def _process_reference(self, path: str):
        """Extract face embedding and save in cache."""
        img = cv2.imread(path)
        if img is None:
            logger.warning("Failed to load image %s", path)
            return
        faces = self.model.get(img)
        if not faces:
            logger.warning("No faces detected in %s", path)
            return
        face = faces[0]
        embedding = face.normed_embedding.tobytes()
        key = path  # could be sha256(file)
        meta = {"type": "face", "source": path}
        self.cache.save_embedding(key, embedding, meta)
        logger.info("Saved embedding for %s", path)
