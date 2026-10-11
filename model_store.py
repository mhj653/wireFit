"""제품 모델 목록 저장소 (모델 하나 = 제품 하나).

    모델      아예 다른 제품이 들어온 것. 이름으로 구분한다.
    체결 포인트 그 제품 **안에서** 체결할 자리. 모델의 하위 개념이다.

그래서 체결 포인트 목록은 모델마다 따로 있다. 모델을 바꾸면 표 내용이 통째로
갈린다.

    Fastening_automation/
      └ Settings/
          └ Models.json     <- 이 파일 하나에 모델 전부

파일 하나에 모아 두는 이유: 프로파일(비전 세팅)처럼 이름을 파일 이름으로 쓰면
모델 이름에 Windows 파일 이름 제약이 걸리고, 목록의 순서를 지킬 수가 없다
(모델 탭은 만든 순서대로 왼쪽부터 쌓인다). 여기서는 이름이 값이라 제약이 없다.

    {
      "version": 1,
      "active": "Model1",
      "models": [
        {"name": "Model1",
         "fastening_points": [{"blob_setting": "...", "calibration": "...", "order": 1}]},
        {"name": "Model2", "fastening_points": []}
      ]
    }

**바꿀 때마다 바로 쓴다**(260910 사용자 결정). 세션 파일(AppSession.json)처럼
종료할 때 몰아 쓰면 강제 종료된 사이에 만든 모델이 사라진다. 모델은 '마지막 UI
상태'가 아니라 제품 정의라서 그러면 안 된다.

읽기 실패는 예외로 올리지 않는다. 모델 파일이 깨졌다고 프로그램이 못 뜨면
현장에서 할 수 있는 일이 없어지므로, 빈 목록으로 돌려주고 화면이 기본 모델을
하나 만들어 이어 간다.
"""

import json
import os

from app_paths import user_data_path
from settings_store import MODELS_FILE_NAME, SETTINGS_DIR_NAME

MODELS_VERSION = 1

# 첫 실행에 만들어 두는 모델 이름. 모델이 하나도 없으면 체결 포인트를 담을
# 자리가 없어서 표를 띄울 수 없다.
DEFAULT_MODEL_NAME = "Model1"

MAX_NAME_LENGTH = 40


def models_path():
    return user_data_path(SETTINGS_DIR_NAME, MODELS_FILE_NAME)


def display_path():
    """상태문구에 띄울 짧은 경로."""
    return os.path.join(SETTINGS_DIR_NAME, MODELS_FILE_NAME)


# ======================================================
# 이름 검사
# ======================================================
def validate_model_name(name, existing_names=()):
    """새 모델 이름이 쓸 만한지 본다.

    파일 이름이 아니라 JSON 안의 값이므로 Windows 제약(\\/:*?"<>|)은 볼 것이
    없다. 대신 **같은 이름이 둘 있으면 안 된다** — 모델 탭과 모델 변경 창이
    이름으로 모델을 가리키므로, 겹치면 어느 것을 고른 것인지 알 수 없다.

    Returns:
        (True, 정리된 이름) 또는 (False, 사유)
    """
    cleaned = (name or "").strip()
    if not cleaned:
        return False, "모델 이름을 입력하세요."
    if len(cleaned) > MAX_NAME_LENGTH:
        return False, f"이름은 {MAX_NAME_LENGTH}자 이내여야 합니다."
    lowered = {str(existing).strip().lower() for existing in existing_names}
    if cleaned.lower() in lowered:
        return False, f"'{cleaned}' 모델이 이미 있습니다."
    return True, cleaned


# ======================================================
# 읽기 / 쓰기
# ======================================================
def _clean_points(value):
    """저장된 체결 포인트 목록을 방어적으로 읽는다.

    파일을 손으로 고쳤을 수 있으므로 dict 가 아닌 항목은 버린다. 빈 값은
    그대로 둔다 — 아직 세팅을 고르지 않은 행도 표에 있어야 한다.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def load_models():
    """저장해 둔 모델 목록을 읽는다. → (모델 목록, 고른 모델 이름)

    모델 목록의 각 항목은 {"name": str, "fastening_points": [dict, ...]} 이다.
    파일이 없거나 깨졌으면 ([], None) 을 돌려준다 — 부르는 쪽이 기본 모델을
    하나 만들어 이어 간다.

    이름이 없거나 겹치는 항목은 버린다. 겹친 이름을 살려 두면 탭과 모델 변경
    창이 어느 모델을 가리키는지 알 수 없다.
    """
    path = models_path()
    if not os.path.exists(path):
        return [], None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, ValueError):
        return [], None
    if not isinstance(data, dict):
        return [], None

    models = []
    seen = set()
    for item in data.get("models") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        models.append({
            "name": name,
            "fastening_points": _clean_points(item.get("fastening_points")),
        })

    active = data.get("active")
    if not isinstance(active, str) or active not in {m["name"] for m in models}:
        active = models[0]["name"] if models else None
    return models, active


def save_models(models, active=None):
    """모델 목록을 통째로 쓴다.

    쓰는 도중 프로그램이 죽어도 기존 파일이 반쯤 덮여 깨지지 않도록 임시
    파일에 먼저 쓰고 통째로 바꿔치운다(settings_store 와 같은 방식).

    Returns:
        (성공 여부, 저장 경로 또는 실패 사유)
    """
    path = models_path()
    payload = {
        "version": MODELS_VERSION,
        "active": active,
        "models": [
            {
                "name": entry.get("name", ""),
                "fastening_points": _clean_points(entry.get("fastening_points")),
            }
            for entry in models or []
        ],
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
        return True, path
    except OSError as exc:
        return False, str(exc)
