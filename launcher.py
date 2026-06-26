import sys
import logging
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from config.settings import Settings
from core.pipeline_manager import PipelineManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-studio")

def run_launcher() -> None:
    """Initialize settings, pipeline and start UI."""
    logger.info("Starting AI Studio")
    settings = Settings.load()
    pipeline = PipelineManager(settings=settings)
    app = QApplication(sys.argv)
    window = MainWindow(pipeline_manager=pipeline, settings=settings)
    window.showMaximized()
    sys.exit(app.exec())
