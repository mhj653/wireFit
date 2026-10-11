"""프로그램 전체에서 동일한 폴더 선택 UI를 사용한다."""

from PyQt5.QtWidgets import QFileDialog


def select_folder(parent, title, initial_path):
    """Windows 기본 폴더 선택창을 열고 선택 경로를 반환한다."""
    return QFileDialog.getExistingDirectory(
        parent,
        title,
        initial_path,
        QFileDialog.ShowDirsOnly,
    )
