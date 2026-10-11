@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -X utf8 -c "import sys; print('python', sys.executable); import torch; print('torch', torch.__version__); import PyQt5.QtCore as qt; print('qt', qt.QT_VERSION_STR); import cv2; print('opencv', cv2.__version__); import pyrealsense2; print('pyrealsense2 OK'); import sam2_detector, sam2_video_tracker; print('sam2 models', sam2_detector.list_models()); print('sam2 image available', sam2_detector.available(), sam2_detector.unavailable_reason()); print('sam2 video available', sam2_video_tracker.available(), sam2_video_tracker.unavailable_reason()); print('sam2 device', sam2_detector.device_note()); import PythonUI; print('PythonUI OK')"
