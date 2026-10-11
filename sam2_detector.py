"""SAM2로 홀을 티칭하고, 이후 촬영에서 같은 홀을 다시 찾는다.

쓰는 방법은 두 단계다.

  1. 티칭 (세팅 화면)
     사람이 홀을 네모로 드래그한다. 그 네모를 SAM2에 프롬프트로 주면 홀 경계
     마스크가 나오고, 거기서 중심과 반지름을 뽑는다. 그 마스크의 외접 사각형이
     '기준 네모'가 되어 이후 촬영의 프롬프트로 쓰인다.

     260914 까지는 점을 클릭했다. 점 하나는 '이 점이 속한 무엇'이라 SAM2 가
     홀인지 판재인지 스스로 골라야 해서, **뜻한 것보다 큰 물체가 잡히는** 일이
     있었다. 네모는 범위를 함께 알려 주므로 뜻한 홀이 그대로 나오고, 네모를
     한참 넘겨 번진 후보는 아예 뺄 수 있다(_hole_picker). 법선 매칭점
     (teach_region)이 처음부터 네모였던 것과 같은 이유다.

  2. 재검출 (오퍼레이터 모드)
     부품이 촬영마다 수~수십 px 흔들리므로 티칭한 좌표를 그대로 쓰면
     빗나간다. 그래서 기준 모양으로 대략 위치를 먼저 찾고(template_matching),
     그 자리에 점 프롬프트를 찍어 홀 경계를 다시 딴다.

왜 이렇게 나눴나
  SAM2의 이미지 인코딩은 입력 크기와 무관하게 1024x1024로 리사이즈해 도는
  고정 비용이라, 이 PC(CPU)에서 촬영 한 장당 약 1.2초가 든다. 잘라 넣어도
  줄지 않는다. 반면 점 프롬프트는 한 번에 40ms 안쪽이다. 그래서 '어디를
  찍을지'는 값싼 템플릿 매칭에 맡기고, SAM2에는 경계를 따는 일만 시킨다.

torch/sam2 가 없어도 이 모듈을 import 하는 쪽은 깨지지 않는다. available()
이 False 를 돌려주고 unavailable_reason() 이 사유를 알려 줄 뿐이다
(카메라 페이지가 pyrealsense2 를 다루는 방식과 같다).
"""

import os
import threading

import cv2
import numpy as np

from app_paths import resource_path

# torch/sam2 가 없거나 못 올라와도 앱의 나머지는 그대로 돌아야 한다.
#
# ImportError 만 잡으면 안 된다. Windows 에서 PyQt5 를 먼저 올린 뒤 torch 를
# 부르면 c10.dll 초기화가 실패해 OSError(WinError 1114) 가 난다. 그것까지
# 잡지 않으면 SAM2 를 안 쓰는 사람의 앱이 통째로 뜨지 않는다.
# (그 순서 문제 자체는 PythonUI.py 가 torch 를 먼저 올려 피한다.)
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception as exc:   # pragma: no cover - 런타임 환경 의존
    torch = None
    build_sam2 = None
    SAM2ImagePredictor = None
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""


# ===== 모델 =====
MODELS_DIR_NAME = os.path.join("models", "sam2")

# 체크포인트 파일 -> sam2 패키지 안의 config 이름. 짝이 맞지 않으면 로드가
# 실패하므로 파일명으로 직접 짝지어 둔다.
MODEL_CONFIGS = {
    "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

# 작은 모델부터. 이 PC는 GPU가 없어 tiny 도 촬영당 1.2초가 드므로 기본은 tiny다.
MODEL_ORDER = [
    "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt",
]

MODEL_LABELS = {
    "sam2.1_hiera_tiny.pt": "tiny (가장 빠름)",
    "sam2.1_hiera_small.pt": "small",
    "sam2.1_hiera_base_plus.pt": "base+ (느림)",
    "sam2.1_hiera_large.pt": "large (가장 느림)",
}

# ===== 마스크 -> 홀 =====
# 프롬프트 한 번에 SAM2는 후보 마스크를 여러 장 준다. 홀은 '작은 구멍'이므로,
# 점수만 보고 고르면 판재 전체가 잡히는 일이 있다. 그래서 넓이가 이 상한을
# 넘는 후보는 홀이 아니라고 보고 뺀다(이미지 넓이 대비 비율).
MAX_HOLE_AREA_RATIO = 0.25

# 마스크가 이보다 작으면 잡티로 본다.
MIN_HOLE_AREA_PX = 20

# 사람이 그린 네모가 곧 '이만한 것을 찾아라'는 뜻이다. 네모를 줘도 SAM2 는
# 후보를 여러 장 내는데 그중에는 네모를 한참 넘겨 판재까지 문 것이 섞인다.
# 이미지 넓이 기준 상한(MAX_HOLE_AREA_RATIO)만으로는 그런 후보가 통과한다 —
# 1280x720 에서는 판재의 4분의 1까지 홀로 인정하는 셈이라, 점 프롬프트 시절
# '원하는 것보다 큰 물체가 잡힌다'던 일이 네모에서도 그대로 일어난다.
# 그래서 네모 자체를 상한으로 함께 쓴다.
HOLE_INSIDE_BOX_RATIO = 0.85   # 마스크 화소의 이만큼은 네모 안에 있어야 한다
HOLE_BOX_AREA_RATIO = 1.2      # 네모 넓이의 이 배를 넘으면 뜻한 홀이 아니다

# ===== 마스크 -> 면 =====
# 법선을 세울 매칭점은 홀과 정반대다. '평평한 면'을 잡는 것이라 홀용
# 상한(MAX_HOLE_AREA_RATIO)으로 거르면 원하는 후보가 먼저 잘려 나간다.
# 그래서 훨씬 너그러운 상한을 따로 둔다. 다만 화면을 다 덮는 후보는 프롬프트가
# 배경에 찍힌 것이므로 뺀다 — 그건 부품 위의 한 자리가 아니다.
MAX_REGION_AREA_RATIO = 0.9
MIN_REGION_AREA_PX = 200


# ===== 연산 장치 =====
# 한 번 정해서 계속 쓴다. 물어볼 때마다 CUDA 를 두드리면 값이 들고, 도중에
# 답이 바뀌면 모델과 텐서가 서로 다른 장치에 놓여 터진다.
_device = None
_device_note = ""
_device_lock = threading.Lock()


def _pick_device():
    """쓸 장치와 그 사유. → (torch.device, 사람이 읽을 한 줄)

    CUDA 가 '있다'고 해서 되는 것이 아니다. 설치된 PyTorch 휠에 이 GPU 세대의
    커널이 들어 있어야 한다. 없으면 모델 로드는 지나가고 **실제 추론에서**
    `no kernel image is available` 로 죽는다.

    이 프로젝트는 그 함정에 정확히 걸린다 — RTX 4050 은 sm_89, RTX 5060 은
    sm_120(Blackwell)이고 sm_120 커널은 CUDA 12.8 이상 휠에만 있다. cu124 휠을
    깔면 4050 에서는 돌고 5060 에서는 죽는다. 그래서 실제로 커널을 한 번
    돌려 보고 정한다. 실패하면 CPU 로 물러서고 사유를 남긴다 — 라인이 멈추는
    것보다 느린 편이 낫고, 문구를 보면 무엇을 깔아야 할지 알 수 있다.
    """
    if torch is None or not torch.cuda.is_available():
        return torch.device("cpu") if torch is not None else None, "CPU (CUDA 없음)"

    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
    except Exception as exc:
        return torch.device("cpu"), f"CPU (GPU 정보를 읽지 못했습니다: {exc})"

    try:
        # 커널이 실제로 있는지 확인한다. 커널이 없으면 여기서 터진다.
        probe = torch.zeros(8, device="cuda")
        probe = probe + 1.0
        torch.cuda.synchronize()
        float(probe.sum())
    except Exception as exc:
        builds = torch.version.cuda or "?"
        return torch.device("cpu"), (
            f"CPU — {name} (sm_{major}{minor}) 에서 쓸 커널이 이 PyTorch 에 "
            f"없습니다 (CUDA {builds} 빌드). sm_120(RTX 50 시리즈)은 CUDA 12.8 "
            f"이상으로 빌드된 PyTorch 가 필요합니다. 상세: {exc}"
        )

    return torch.device("cuda"), f"CUDA — {name} (sm_{major}{minor})"


def device():
    """SAM2 를 올릴 장치. 처음 부를 때 한 번 정한다."""
    global _device, _device_note
    with _device_lock:
        if _device is None:
            _device, _device_note = _pick_device()
        return _device


def device_note():
    """어떤 장치를 왜 쓰는지 (사람이 읽을 한 줄). 화면과 로그에 띄운다."""
    device()
    return _device_note


def using_cuda():
    picked = device()
    return picked is not None and picked.type == "cuda"


def available():
    """SAM2를 쓸 수 있는 환경인지."""
    return torch is not None and bool(list_models())


def unavailable_reason():
    """쓸 수 없다면 그 사유 (사람이 읽을 한 줄). 쓸 수 있으면 빈 문자열."""
    if torch is None:
        return f"PyTorch/sam2가 설치되어 있지 않습니다 ({_IMPORT_ERROR})"
    if not list_models():
        return f"{MODELS_DIR_NAME} 폴더에 SAM2 체크포인트(.pt)가 없습니다"
    return ""


def models_dir():
    """체크포인트 폴더. exe로 묶어도 같은 자리를 가리킨다."""
    return resource_path(MODELS_DIR_NAME)


def list_models():
    """쓸 수 있는 체크포인트 파일명 목록 (작은 모델 먼저)."""
    folder = models_dir()
    try:
        names = set(os.listdir(folder))
    except OSError:
        return []
    return [
        name
        for name in MODEL_ORDER
        if name in names and os.path.isfile(os.path.join(folder, name))
    ]


def model_label(name):
    """콤보에 띄울 이름."""
    return MODEL_LABELS.get(name, name)


def default_model():
    """기본으로 고를 체크포인트. 없으면 None."""
    models = list_models()
    return models[0] if models else None


# ======================================================
# 모델 보관
# ======================================================
class _Segmenter:
    """체크포인트 하나를 물고 있는 SAM2 예측기.

    모델 로드가 1초 가까이 걸리므로 한 번 올린 것을 계속 쓴다. 같은 이미지를
    거듭 프롬프트할 때 인코딩(약 1.2초)도 다시 하지 않도록, 마지막으로 인코딩한
    이미지를 내용으로 기억해 둔다.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self._predictor = None
        self._encoded_key = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._predictor is not None:
            return
        config = MODEL_CONFIGS.get(self.model_name)
        if config is None:
            raise RuntimeError(f"모르는 SAM2 체크포인트입니다: {self.model_name}")
        path = os.path.join(models_dir(), self.model_name)
        if not os.path.isfile(path):
            raise RuntimeError(f"SAM2 체크포인트를 찾을 수 없습니다: {path}")
        picked = device()
        if picked.type == "cpu":
            # CPU 스레드를 다 쓰게 둔다. GPU 가 없으면 그래도 초 단위다.
            torch.set_num_threads(os.cpu_count() or 4)
        model = build_sam2(config, path, device=str(picked))
        self._predictor = SAM2ImagePredictor(model)

    @staticmethod
    def _image_key(bgr_image):
        """같은 이미지인지 값싸게 가리는 열쇠.

        전체를 해시하면 그것대로 비싸므로, 크기와 드문드문 뽑은 표본만 본다.
        틀려도 '다시 인코딩'일 뿐 결과가 어긋나지는 않는다.
        """
        sample = bgr_image[::37, ::41]
        return (bgr_image.shape, int(sample.sum()), int(sample[..., 0].std() * 1000))

    def segment_box(self, bgr_image, box, picker=None):
        """드래그한 네모를 프롬프트로 주고 그 안의 대상 경계를 딴다.

        홀도 면도 모두 이 길로 딴다(260914). 점 프롬프트는 '이 점이 속한 무엇'
        이라 SAM2가 홀인지 면인지 부품 전체인지 스스로 골라야 하지만, 네모는
        범위를 함께 알려 주므로 사람이 뜻한 영역이 그대로 나온다.

        Args:
            bgr_image: 원본 BGR 이미지
            box: (left, top, right, bottom) 원본 이미지 픽셀
            picker: 후보 마스크 중 하나를 고르는 함수 (masks, scores, shape).
                기본은 면(제외 영역)용이다. 홀은 크기 기준이 달라
                _hole_picker(box) 를 넣는다.

        Returns:
            (H, W) bool 마스크 또는 None
        """
        left, top, right, bottom = (float(value) for value in box)
        with self._lock:
            self._ensure_loaded()
            key = self._image_key(bgr_image)
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            with torch.inference_mode():
                # 같은 이미지면 인코딩(1.2초)을 건너뛴다. 홀과 매칭점을 한
                # 촬영에서 잇달아 딸 때 두 번째가 40ms 안쪽으로 끝나는 이유다.
                if key != self._encoded_key:
                    self._predictor.set_image(rgb)
                    self._encoded_key = key
                masks, scores, _logits = self._predictor.predict(
                    box=np.array([left, top, right, bottom]),
                    multimask_output=True,
                )
        choose = picker or _pick_region_mask
        return choose(masks, scores, bgr_image.shape[:2])


_segmenters = {}
_segmenters_lock = threading.Lock()


def segmenter(model_name=None):
    """체크포인트별로 하나씩만 만들어 돌려쓴다."""
    name = model_name or default_model()
    if name is None:
        raise RuntimeError(unavailable_reason() or "쓸 수 있는 SAM2 모델이 없습니다.")
    with _segmenters_lock:
        if name not in _segmenters:
            _segmenters[name] = _Segmenter(name)
        return _segmenters[name]


def _hole_picker(box):
    """드래그한 네모에 맞춰 '홀' 후보를 고르는 함수를 만든다.

    점수가 가장 높은 것이 늘 홀은 아니다. 판재 위의 작은 구멍을 프롬프트로 줘도
    판재 전체가 더 그럴듯한 분할로 나오기도 한다. 그래서 둘을 함께 본다.

      1. 네모 밖으로 번진 후보를 뺀다 — 마스크 화소의 대부분이 네모 안에 있어야
         한다. 사람이 홀을 감싸 그린 네모가 곧 '이만한 것'이라는 뜻이다.
      2. 네모(와 이미지)보다 넓은 후보를 뺀다.

    남은 것 중 점수가 가장 높은 것을 쓴다. 하나도 남지 않으면 None 이고,
    부르는 쪽(teach)이 '다시 그려 보라'고 알린다 — 여기서 조건을 풀어 큰 것을
    돌려주면 사람이 뜻한 적 없는 물체가 기준 홀로 저장된다.
    """
    left, top, right, bottom = (int(round(float(value))) for value in box)
    left, top = max(0, left), max(0, top)
    box_area = max(1, (right - left) * (bottom - top))

    def pick(masks, scores, image_shape):
        height, width = image_shape
        limit = min(
            height * width * MAX_HOLE_AREA_RATIO, box_area * HOLE_BOX_AREA_RATIO
        )
        best = None
        best_score = -1.0
        for mask, score in zip(masks, scores):
            mask = np.asarray(mask, dtype=bool)
            area = int(mask.sum())
            if area < MIN_HOLE_AREA_PX or area > limit:
                continue
            inside = int(mask[top:bottom, left:right].sum())
            if inside < area * HOLE_INSIDE_BOX_RATIO:
                continue
            if float(score) > best_score:
                best, best_score = mask, float(score)
        return best

    return pick


def _pick_region_mask(masks, scores, image_shape):
    """후보 마스크 중 '제외 영역'으로 볼 만한 것을 고른다.

    홀과 반대로 넓은 쪽이 정답이다. 점수가 가장 높은 것을 쓰되, 화면을 거의
    다 덮는 후보만 뺀다 — 그건 프롬프트가 배경에 찍혔다는 뜻이고, 그대로
    제외하면 법선을 뽑을 뎁스가 남지 않는다.
    """
    height, width = image_shape
    limit = height * width * MAX_REGION_AREA_RATIO

    best = None
    best_score = -1.0
    for mask, score in zip(masks, scores):
        mask = np.asarray(mask, dtype=bool)
        area = int(mask.sum())
        if area < MIN_REGION_AREA_PX or area > limit:
            continue
        if float(score) > best_score:
            best, best_score = mask, float(score)
    return best


def mask_bounds(mask, min_area=MIN_REGION_AREA_PX):
    """마스크에서 가장 큰 덩어리의 (left, top, right, bottom). 없으면 None.

    조각이 여럿이면 가장 큰 것만 쓴다. 멀리 떨어진 잡티까지 담으면 기준 모양이
    쓸데없이 커져 대략 위치 찾기가 무뎌진다.

    min_area 를 받는 이유: 기본값은 면(법선 매칭점)의 하한이라 홀에는 너무
    크다. 홀은 반지름 8px 만 돼도 200px 을 못 채우는데, 그러면 mask_to_hole
    (하한 20px)은 홀을 찾아 놓고 여기서 네모만 못 잡는 일이 생긴다. 홀을 딸
    때는 MIN_HOLE_AREA_PX 를 넘긴다.
    """
    if mask is None:
        return None
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[biggest, cv2.CC_STAT_AREA]) < min_area:
        return None
    left = int(stats[biggest, cv2.CC_STAT_LEFT])
    top = int(stats[biggest, cv2.CC_STAT_TOP])
    return (
        left,
        top,
        left + int(stats[biggest, cv2.CC_STAT_WIDTH]),
        top + int(stats[biggest, cv2.CC_STAT_HEIGHT]),
    )


def mask_to_hole(mask):
    """마스크에서 (x, y, radius)를 뽑는다. 쓸 수 없으면 None.

    조각이 여럿이면 가장 큰 덩어리만 쓴다. 프롬프트 한 점 주변의 홀이
    목적이므로, 멀리 떨어진 잡티까지 중심 계산에 넣으면 안 된다.

    반지름은 넓이에서 되돌린 '같은 넓이의 원'의 반지름이다. 홀이 비스듬히
    보여 타원이 되어도 한 값으로 안정적으로 나온다.
    """
    if mask is None:
        return None
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None
    # 0번은 배경이므로 뺀다.
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[biggest, cv2.CC_STAT_AREA])
    if area < MIN_HOLE_AREA_PX:
        return None
    cx, cy = centroids[biggest]
    return (int(round(cx)), int(round(cy)), float(np.sqrt(area / np.pi)))


# ======================================================
# 티칭
# ======================================================
def teach(bgr_image, box, model_name=None):
    """드래그한 네모 안의 홀을 티칭한다.

    법선 매칭점(teach_region)과 **프롬프트도 결과도 같다.** 둘 다 '이 네모 안의
    것을 다시 찾아라'가 되므로 저장 구조와 재검출 경로를 하나로 쓸 수 있다.
    다른 점은 홀은 마스크에서 중심과 반지름까지 뽑는다는 것뿐이다.

    크롭(template.png)을 더 만들지 않는다. 그것은 cv2.matchTemplate 으로 대략
    위치를 잡던 2단계 방식이 쓰던 것인데, 그 방식은 배율도 회전도 보지 않아
    카메라 거리가 20%만 바뀌어도 판재를 물고 **그럴듯한 오답**을 냈다
    (260908 실측: 축소 0.8 에서 반지름 20px 이 나와야 할 자리에 257px).
    못 찾는 것보다 나쁘므로 걷어냈다.

    Args:
        bgr_image: 원본 BGR 이미지
        box: (left, top, right, bottom) 드래그한 네모 (원본 이미지 픽셀)
        model_name: 쓸 체크포인트. None 이면 기본값.

    Returns:
        (hole, mask, taught_box) — hole 은 (x, y, radius),
        mask 는 (H, W) bool, taught_box 는 그 마스크의 외접 사각형
        (left, top, right, bottom). 다음 촬영에서 이 네모가 프롬프트가 된다.

        사람이 그린 네모가 아니라 SAM2 가 실제로 잡은 경계의 외접 사각형을
        돌려주는 것도 teach_region 과 같다 — 사람 손은 넉넉하게 그리기 마련이고,
        그 여백이 넓으면 다음 촬영에서 옆 물체까지 프롬프트 안에 들어온다.

    Raises:
        RuntimeError: 그 네모에서 홀을 찾지 못했을 때
    """
    mask = segmenter(model_name).segment_box(
        bgr_image, box, picker=_hole_picker(box)
    )
    hole = mask_to_hole(mask)
    # 두 값이 같은 덩어리·같은 하한에서 나오므로 둘 중 하나만 None 이 되는
    # 일은 없다. 한 문장으로 알린다 — 사람이 할 일은 어느 쪽이든 같다.
    bounds = mask_bounds(mask, min_area=MIN_HOLE_AREA_PX)
    if hole is None or bounds is None:
        raise RuntimeError(
            "네모 안에서 홀을 찾지 못했습니다. 홀 하나만 네모 안에 들어오게 "
            "다시 그려 보세요."
        )
    return hole, mask, tuple(int(value) for value in bounds)


def teach_region(bgr_image, box, model_name=None):
    """드래그한 네모 안의 면을 티칭한다 (법선 매칭점 한 곳).

    홀 티칭(teach)과 프롬프트가 같다. 다른 것은 고르는 기준뿐이다 — 여기서는
    넓은 쪽이 정답이라 _pick_region_mask 를 쓴다.

    경계 마스크를 원으로 줄이지 않는다. 짚은 자리는 원이 아니라 면이다.

    Args:
        bgr_image: 원본 BGR 이미지
        box: (left, top, right, bottom) 드래그한 네모 (원본 이미지 픽셀)
        model_name: 쓸 체크포인트. None 이면 기본값.

    Returns:
        (mask, taught_box) — mask 는 (H, W) bool,
        taught_box 는 실제로 쓸 (left, top, right, bottom) 정수 네모.

        사람이 그린 네모가 아니라 SAM2 가 실제로 잡은 경계의 외접 사각형을
        돌려준다. 사람 손은 넉넉하게 그리기 마련이고, 그 여백이 넓으면 다음
        촬영에서 옆 물체까지 프롬프트 안에 들어온다.

    Raises:
        RuntimeError: 그 네모에서 영역을 잡지 못했을 때
    """
    mask = segmenter(model_name).segment_box(
        bgr_image, box, picker=_pick_region_mask
    )
    bounds = mask_bounds(mask)
    if bounds is None:
        raise RuntimeError(
            "지정한 네모에서 면을 잡지 못했습니다. "
            "평평한 자리가 네모 안에 넉넉히 들어오게 다시 그려 보세요."
        )
    return mask, tuple(int(value) for value in bounds)
