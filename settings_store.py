"""세팅 파일 저장/불러오기 (체결 위치별로 여러 벌).

체결 위치마다 블롭 파라미터와 ROI가 다르므로, 세팅을 '프로파일' 단위로 나눠
각각 별도의 JSON 파일에 담는다. 프로파일 이름이 곧 파일 이름이다.

    Fastening_automation/
      └ Settings/
          ├ BlobSetting.json        <- 기본 프로파일
          ├ 좌측상단홀.json          <- 사용자가 추가한 것
          └ ActiveSetting.json      <- 마지막으로 고른 프로파일 (목록에는 안 뜸)

저장 위치는 실행 위치와 무관하게 항상 이 파일이 있는 폴더 기준이다
(바탕화면에서 실행하든 IDE에서 실행하든 같은 파일을 본다).

페이지는 이 모듈만 부르면 되고, 폴더 생성·깨진 파일·이름 검사는 여기서 맡는다.
"""

import json
import os

from app_paths import user_data_path

SETTINGS_DIR_NAME = "Settings"
DEFAULT_PROFILE_NAME = "BlobSetting"

# 마지막으로 고른 프로파일을 기억하는 파일. 프로파일 목록에는 나오지 않는다.
ACTIVE_FILE_STEM = "ActiveSetting"
ACTIVE_FILE_NAME = ACTIVE_FILE_STEM + ".json"
SESSION_FILE_NAME = "AppSession.json"
# 제품 모델 목록 (model_store 가 쓴다). 같은 Settings 폴더에 있으므로 여기서도
# 알아야 한다 — 안 그러면 비전 세팅 목록에 'Models' 가 프로파일처럼 뜬다.
MODELS_FILE_NAME = "Models.json"

# 이 폴더에서 프로그램이 쓰는 파일들. 프로파일 목록에서 빼고, 사용자가 같은
# 이름으로 프로파일을 만들지도 못하게 한다.
RESERVED_FILE_NAMES = (ACTIVE_FILE_NAME, SESSION_FILE_NAME, MODELS_FILE_NAME)
RESERVED_FILE_STEMS = {
    os.path.splitext(name)[0].lower() for name in RESERVED_FILE_NAMES
}

# 저장 형식이 나중에 바뀔 때 구분하려고 남겨 둔다.
SETTINGS_VERSION = 1

# 이름 검사 규칙 (파일 이름으로 그대로 쓰이므로 Windows 제약을 따른다)
INVALID_NAME_CHARS = '\\/:*?"<>|'
MAX_NAME_LENGTH = 40
RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


# ======================================================
# 경로
# ======================================================
def settings_dir():
    return user_data_path(SETTINGS_DIR_NAME)


def profile_path(name):
    return os.path.join(settings_dir(), f"{name}.json")


def display_path(path):
    """상태문구에 띄울 짧은 경로 (Settings/이름.json)."""
    return os.path.join(SETTINGS_DIR_NAME, os.path.basename(path))


# ======================================================
# 프로파일 목록
# ======================================================
def list_profiles():
    """저장된 프로파일 이름을 가나다순으로 돌려준다."""
    try:
        names = os.listdir(settings_dir())
    except OSError:
        return []
    return sorted(
        os.path.splitext(name)[0]
        for name in names
        if name.lower().endswith(".json")
        and name not in RESERVED_FILE_NAMES
    )


def profile_exists(name):
    return os.path.exists(profile_path(name))


def validate_profile_name(name):
    """새 프로파일 이름이 쓸 만한지 본다.

    Returns:
        (True, 정리된 이름) 또는 (False, 사유)
    """
    cleaned = (name or "").strip()
    if not cleaned:
        return False, "이름을 입력하세요."
    if len(cleaned) > MAX_NAME_LENGTH:
        return False, f"이름은 {MAX_NAME_LENGTH}자 이내여야 합니다."

    used = sorted({char for char in cleaned if char in INVALID_NAME_CHARS})
    if used:
        return False, "파일 이름에 쓸 수 없는 문자입니다: " + " ".join(used)
    # 앞뒤 공백은 위에서 이미 떼어냈으므로 마침표만 보면 된다.
    # (Windows 는 마침표로 끝나는 파일 이름을 만들지 못한다.)
    if cleaned.endswith("."):
        return False, "이름 끝에 마침표는 쓸 수 없습니다."
    if cleaned.upper() in RESERVED_NAMES:
        return False, f"'{cleaned}' 은(는) Windows 예약어라 쓸 수 없습니다."
    if cleaned.lower() in RESERVED_FILE_STEMS:
        return False, f"'{cleaned}' 은(는) 프로그램이 쓰는 이름이라 쓸 수 없습니다."

    # Windows 는 파일 이름 대소문자를 구분하지 않으므로 같은 것으로 본다.
    if cleaned.lower() in {existing.lower() for existing in list_profiles()}:
        return False, f"'{cleaned}' 세팅이 이미 있습니다."
    return True, cleaned


# ======================================================
# 읽기 / 쓰기 / 삭제
# ======================================================
def save_profile(name, data):
    """프로파일 하나를 저장한다.

    쓰는 도중 프로그램이 죽어도 기존 파일이 반쯤 덮여 깨지지 않도록
    임시 파일에 먼저 쓰고 통째로 바꿔치운다.

    Returns:
        (성공 여부, 저장 경로 또는 실패 사유)
    """
    path = profile_path(name)
    payload = dict(data)
    payload["version"] = SETTINGS_VERSION
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
        return True, path
    except OSError as exc:
        return False, str(exc)


def load_profile(name):
    """프로파일 하나를 읽는다.

    파일이 없거나(첫 실행) 내용이 깨졌으면 None을 돌려준다. 세팅이 없다고
    프로그램이 못 뜨면 안 되므로, 읽기 실패는 예외 대신 None으로 알린다.
    """
    path = profile_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def delete_profile(name):
    """프로파일 파일을 지운다.

    Returns:
        (성공 여부, None 또는 실패 사유)
    """
    try:
        os.remove(profile_path(name))
        return True, None
    except OSError as exc:
        return False, str(exc)


# ======================================================
# 마지막으로 고른 프로파일
# ======================================================
def active_profile():
    """마지막으로 고른 프로파일 이름. 기록이 없거나 그 사이 지워졌으면 None."""
    path = os.path.join(settings_dir(), ACTIVE_FILE_NAME)
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("active")
    # 기록해 둔 프로파일이 그 사이 삭제됐을 수 있으므로 실제 파일을 확인한다.
    if isinstance(name, str) and profile_exists(name):
        return name
    return None


def set_active_profile(name):
    """다음 실행에서 이 프로파일로 시작하도록 기록한다. 실패해도 조용히 넘어간다."""
    path = os.path.join(settings_dir(), ACTIVE_FILE_NAME)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump({"active": name}, fp, ensure_ascii=False, indent=2)
    except OSError:
        # 어떤 프로파일로 시작할지는 없어도 되는 정보다(첫 항목으로 시작하면 된다).
        pass
