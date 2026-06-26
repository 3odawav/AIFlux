from __future__ import annotations
import logging
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QSplitter, QLabel, QPushButton
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction
from core.pipeline_manager import PipelineManager
from config.settings import Settings
from ui.identity_page import IdentityPage

logger = logging.getLogger("ui")

class MainWindow(QMainWindow):
    def __init__(self, pipeline_manager: PipelineManager, settings: Settings):
        super().__init__()
        self.pipeline = pipeline_manager
        self.settings = settings
        self.setWindowTitle("AI Studio Pro")
        self.resize(1400, 900)
        self._init_ui()

    def _init_ui(self):
        # Left sidebar
        sidebar = QListWidget()
        for item in ["Home","Models","Identity","Outfit","Background","Camera","Lighting","Style","ControlNet","LoRA","Performance","Gallery","History","Settings"]:
            sidebar.addItem(item)
        sidebar.setFixedWidth(180)

        # Center main canvas
        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.preview_label = QLabel("Image Preview Area")
        self.preview_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.preview_label)

        # Right panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Prompt"))
        right_layout.addWidget(QLabel("Prompt input will be here"))
        generate_btn = QPushButton("Generate")
        generate_btn.clicked.connect(self.on_generate)
        right_layout.addWidget(generate_btn)

        # identity page dock (switchable)
        self.identity_page = IdentityPage(self.pipeline)

        # main splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    def on_generate(self):
        # Example: create a request and send to pipeline
        from core.pipeline_manager import GenerateRequest
        req = GenerateRequest(prompt="A portrait of a man in studio lighting", steps=28)
        job_id = self.pipeline.generate(req)
        logger.info("Submitted job %s", job_id)
