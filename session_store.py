"""프로그램을 다시 실행할 때 복원할 UI 세션 상태."""

import json
import os

from app_paths import user_data_path


SESSION_FILE_NAME = "AppSession.json"
SESSION_PATH = user_data_path("Settings", SESSION_FILE_NAME)
SESSION_VERSION = 1


def load_session():
    try:
        with open(SESSION_PATH, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_session(data):
    payload = dict(data)
    payload["version"] = SESSION_VERSION
    temp_path = SESSION_PATH + ".tmp"
    try:
        os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        os.replace(temp_path, SESSION_PATH)
        return True
    except OSError:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return False
