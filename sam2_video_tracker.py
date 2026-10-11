"""SAM2 비디오 예측기로 티칭한 홀을 추적한다.

티칭한 것을 새 촬영에서 다시 찾는 유일한 경로다.

    260908 이전에는 '2단계' 방식도 있었다. cv2.matchTemplate 으로 크롭을 찾아
    대략 위치를 잡고 SAM2 가 그 자리에서 경계만 따는 것이었는데, 매칭이 배율도
    회전도 보지 않아 카메라 거리가 20%만 바뀌면 엉뚱한 자리를 짚었다. 그러면
    SAM2 가 거기서 판재를 물어 **그럴듯한 오답**이 나온다(실측: 반지름 20px 이
    나와야 할 자리에 257px, 넓이 상한 검사도 통과). 못 찾는 것보다 나쁘므로
    걷어냈다.

    지금 방식은 티칭 프레임과 현재 프레임을 '2장짜리 비디오'로 묶어 비디오
    예측기에 넘긴다. 어디를 볼지도 SAM2 가 정한다. 메모리 어텐션이 모양을
    기억하므로 거리와 회전이 바뀌어도 따라간다.

왜 하필 2장인가
    비디오 예측기는 보통 앞 프레임을 기억하며 이어 달리지만, 그러면 촬영을
    거듭할수록 오차가 쌓인다(드리프트). 프레임을 딱 두 장 — 티칭한 것과 지금
    것 — 으로 두면 현재 프레임이 참조하는 메모리에는 티칭 프레임 하나뿐이다.
    장시간 돌려도 기준이 흔들리지 않고, 촬영끼리 서로 영향을 주지도 않는다.

값은 얼마나 드나 (이 PC, CPU, tiny)
    모델 로드 1.2초와 티칭 프레임 인코딩 1.6초는 각각 한 번뿐이라 캐시해 둔다.
    촬영마다 드는 것은 현재 프레임 인코딩과 메모리 어텐션으로 약 2초다.

    init_state() 가 비디오 파일이나 JPEG 폴더만 받는 탓에 예전에는 이 방식을
    '촬영마다 파일로 쓰고 두 장을 다시 인코딩해야 해서 3초 이상'이라고 보고
    접었다. 실제로는 inference_state 가 평범한 dict 이고 images 가 텐서 스택일
    뿐이라, 파일을 거치지 않고 frame 1 만 갈아끼울 수 있다. 그래서 티칭 프레임
    인코딩이 촬영마다 되풀이되지 않는다.
"""

import contextlib
import os
import threading
from collections import OrderedDict

import cv2
import numpy as np

import sam2_detector
from sam2_detector import (
    MAX_HOLE_AREA_RATIO,
    MAX_REGION_AREA_RATIO,
    mask_to_hole,
)

# sam2_detector 와 같은 이유로 넓게 잡는다. torch 가 없거나 못 올라와도 이
# 모듈을 import 하는 쪽은 깨지지 않아야 한다.
try:
    import torch
    from sam2.build_sam import build_sam2_video_predictor
except Exception as exc:   # pragma: no cover - 런타임 환경 의존
    torch = None
    build_sam2_video_predictor = None
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""

# SAM2 가 쓰는 정규화 상수 (sam2/utils/misc.py 의 load_video_frames 기본값).
# 여기 값이 어긋나면 티칭 프레임과 현재 프레임이 서로 다른 척도로 인코딩되어
# 추적이 조용히 나빠진다.
_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD = (0.229, 0.224, 0.225)

# 추적한 마스크가 이보다 크면 홀이 아니라 판재를 문 것으로 본다.
# 추적이 어긋났을 때 엉뚱한 좌표를 내놓는 것보다 '못 찾음'이 낫다.
_MAX_AREA_RATIO = MAX_HOLE_AREA_RATIO

# 평면을 세우는 점의 개수. **셋으로 고정이고, 넷 이상을 받지 않는다.**
#
# 둘이면 축 하나가 자유롭게 돌아 법선이 정해지지 않는다. 반대로 넷 이상을
# 최소제곱으로 맞추면 '네 점이 모두 같은 평면에 있다'를 전제하게 되는데,
# 사람이 짚은 자리가 실제로 같은 면인지는 보장할 수 없다. 한 점이 다른 단차에
# 올라가 있어도 잔차로 흩어 놓고 그럴듯한 법선을 내놓아, 틀린 것을 알아챌
# 방법이 없어진다. 셋이면 평면이 유일하게 정해지고 근사가 끼어들지 않는다.
NORMAL_POINTS = 3

# ===== 매칭점이 '제대로 잡혔나'를 재는 기준 =====
# **티칭했을 때의 넓이에 견준다.** 절대 픽셀 수로 재면 안 된다 — 매칭점은 사람이
# 크기를 정하는 것이라, 얼마가 정상인지는 그 점마다 다르다.
#
# 260908 실측이 그 값을 치렀다. 절대 문턱 200px 을 쓰던 때, 티칭 넓이 204px 인
# 점이 추적에서 184px 로 나와 **16px 차이로 매번 탈락**했다. 추적은 멀쩡히 그
# 자리를 찾고 있었는데도 그랬다. 세 곳이 다 있어야 평면이 서므로 그 점 하나가
# 촬영마다 법선을 통째로 죽였다.
#
# 비율을 넉넉히 두는 이유 — **틀린 거절이 틀린 통과보다 훨씬 나쁘다.**
#   거절하면 세 점 중 하나가 빠져 평면이 통째로 죽고, 법선이 홀 고리로 물러선다
#   (260901 실측 흔들림 ±8도). 통과시키면 마스크가 좀 작을 뿐 여전히 같은 면
#   위에 있어서 뎁스 중앙값은 멀쩡하다 — 대표점이 거의 안 움직인다.
#   그래서 여기서 잡으려는 것은 '조금 줄어든 것'이 아니라 **파국적 붕괴**다.
#
# 넓이는 거리의 **제곱**에 반비례하므로 이 값이 곧 거리 여유가 된다.
#   0.50 이면 거리 1.4배까지 (실측: 1.43배에서 48~52%로 탈락했다)
#   0.25 이면 거리 2.0배까지
# 260908 사용자 결정: 0.25. 티칭한 자세에서 벗어나 찍어도 살아남게 둔다.
NORMAL_POINT_MIN_AREA_RATIO = 0.25

# 다만 아래로 끝없이 열어 둘 수는 없다. 이 마스크에서 뎁스 표본을 뽑아 대표점을
# 세우는데(normal_plane.point_from_mask), 거기에 유효 뎁스 화소가
# MIN_POINT_SAMPLES(=10) 개는 있어야 한다. 뎁스가 비는 화소를 감안해 네 배쯤
# 여유를 둔다. 예전의 200px 과 달리 이 값은 '통계가 성립하는 최소'라는 근거가 있다.
NORMAL_POINT_MIN_AREA_PX = 40


def _min_area_for(taught_area):
    """이 매칭점이 통과해야 할 최소 넓이 (px).

    티칭 때 넓이를 모르면(옛 자산) 절대 하한만 쓴다. 그때 예전의 200px 으로
    돌아가지 않는 이유: 그 값이 애초에 멀쩡한 점을 떨어뜨리던 장본인이다.
    """
    if not taught_area:
        return NORMAL_POINT_MIN_AREA_PX
    return max(NORMAL_POINT_MIN_AREA_PX, int(taught_area * NORMAL_POINT_MIN_AREA_RATIO))


def measure_hole_area(scene_bgr, box, model_name=None):
    """티칭 프레임 자신을 추적해 홀의 기준 넓이를 잰다. 못 잡으면 0.

    measure_normal_point_areas 와 같은 이유다 — 재는 자를 맞춘다. 티칭은 이미지
    예측기, 재검출은 비디오 예측기라 마스크 넓이가 서로 다르므로, 비교할 값은
    비교할 방법으로 재야 한다.
    """
    return measure_normal_point_areas(scene_bgr, [box], model_name)[0]


def measure_normal_point_areas(scene_bgr, boxes, model_name=None):
    """티칭 프레임 자신을 추적해 매칭점 세 곳의 기준 넓이를 잰다.

    **재는 자를 맞추려는 것이다.** 티칭은 이미지 예측기(segment_box)로 하고
    재검출은 비디오 예측기로 하는데, 둘이 내놓는 마스크 넓이가 다르다. 실측에서
    같은 자리가 이미지 예측기로 3613px, 비디오 추적으로 174px 였다 — 티칭 쪽
    값을 기준으로 삼으면 문턱이 20배로 부풀어 그 점은 영영 못 쓰게 된다.

    덤으로 '티칭은 됐는데 추적은 못 하는' 자리를 저장 시점에 걸러낸다. 넓이 0 은
    비디오 추적이 그 자리를 아예 못 잡았다는 뜻이고, 그대로 저장하면 촬영마다
    실패한다.

    Returns:
        [넓이] 세 개. 못 잡은 자리는 0.
    """
    scene = _taught_scene(model_name, scene_bgr, tuple(("box", box) for box in boxes))
    return [0 if mask is None else int(mask.sum()) for mask in scene.track(scene_bgr)]


def available():
    """비디오 추적을 쓸 수 있는 환경인지."""
    return (
        torch is not None
        and build_sam2_video_predictor is not None
        and bool(sam2_detector.list_models())
    )


def unavailable_reason():
    """쓸 수 없다면 그 사유 (사람이 읽을 한 줄). 쓸 수 있으면 빈 문자열."""
    if torch is None or build_sam2_video_predictor is None:
        return f"PyTorch/sam2가 설치되어 있지 않습니다 ({_IMPORT_ERROR})"
    if not sam2_detector.list_models():
        return f"{sam2_detector.MODELS_DIR_NAME} 폴더에 SAM2 체크포인트(.pt)가 없습니다"
    return ""


# ======================================================
# 예측기 보관
# ======================================================
_predictors = {}
_predictors_lock = threading.Lock()


def _predictor(model_name):
    """체크포인트별로 하나씩만 만들어 돌려쓴다. 로드가 1.2초쯤 걸린다."""
    name = model_name or sam2_detector.default_model()
    if name is None:
        raise RuntimeError(unavailable_reason() or "쓸 수 있는 SAM2 모델이 없습니다.")
    with _predictors_lock:
        if name not in _predictors:
            config = sam2_detector.MODEL_CONFIGS.get(name)
            if config is None:
                raise RuntimeError(f"모르는 SAM2 체크포인트입니다: {name}")
            path = os.path.join(sam2_detector.models_dir(), name)
            if not os.path.isfile(path):
                raise RuntimeError(f"SAM2 체크포인트를 찾을 수 없습니다: {path}")
            picked = sam2_detector.device()
            if picked.type == "cpu":
                torch.set_num_threads(os.cpu_count() or 4)
            else:
                # Ada / Blackwell 둘 다 TF32 를 지원한다. 행렬곱이 빨라지고
                # 마스크 경계에 영향을 줄 정밀도는 아니다.
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            _predictors[name] = build_sam2_video_predictor(
                config, path, device=str(picked)
            )
        return _predictors[name]


def _inference_context():
    """추론을 감쌀 컨텍스트. CUDA 면 bfloat16 autocast 를 함께 켠다.

    SAM2 가 권하는 방식이다. 4050(Ada)과 5060(Blackwell) 모두 bfloat16 을
    하드웨어로 지원한다. CPU 에서는 autocast 가 오히려 느려질 수 있어 켜지 않는다.
    """
    if sam2_detector.using_cuda():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _to_tensor(bgr_image, image_size):
    """BGR 한 장을 SAM2 입력 텐서로. init_state 가 파일에 하는 일과 같다."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(_IMG_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(_IMG_STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


class _TaughtScene:
    """티칭 프레임을 frame 0 으로 물고 있는 추론 상태.

    frame 0 의 인코딩과 조건부 메모리는 한 번 만들면 바뀌지 않으므로 계속 쓴다.
    촬영마다 하는 일은 frame 1 을 갈아끼우고 frame 1 의 추적 결과만 지우는
    것뿐이다.

    한 상태가 대상 여럿을 함께 물 수 있다. 법선을 뽑는 매칭점이 셋 이상이고
    그것들이 모두 같은 티칭 프레임에 있으므로, 따로 돌리면 촬영마다 프레임
    인코딩(약 1.5초)을 점 수만큼 되풀이하게 된다. 한 상태에 객체로 넣으면
    인코딩을 한 번만 하고 대상마다 마스크 디코딩만 더 든다.
    """

    def __init__(self, model_name, scene_bgr, prompts):
        """
        Args:
            prompts: (종류, 좌표) 튜플들. 종류는 "point" 또는 "box".
                point 는 (x, y), box 는 (left, top, right, bottom).
        """
        self._model_name = model_name
        self._scene = scene_bgr
        self._prompts = tuple(prompts)
        self._state = None
        self._predictor_ref = None
        self._lock = threading.Lock()

    @property
    def scene_shape(self):
        return self._scene.shape[:2]

    def _ensure_ready(self):
        """티칭 프레임을 인코딩하고 프롬프트를 걸어 둔다 (1회, 약 1.6초)."""
        if self._state is not None:
            return
        predictor = _predictor(self._model_name)
        picked = sam2_detector.device()
        height, width = self._scene.shape[:2]
        frame = _to_tensor(self._scene, predictor.image_size)
        # frame 1 자리는 촬영마다 덮어쓴다. 일단 티칭 프레임으로 채워 둔다.
        # 프레임이 둘뿐이라 GPU 에 올려 둬도 24MB 남짓이다. 촬영마다 옮기는
        # 것보다 낫다.
        state = {
            "images": torch.stack([frame, frame.clone()], dim=0).to(picked),
            "num_frames": 2,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": height,
            "video_width": width,
            "device": picked,
            "storage_device": picked,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        with torch.inference_mode(), _inference_context():
            # obj_id 를 1부터 차례로 매겨 prompts 의 차례와 맞춘다. 나중에
            # propagate 가 주는 마스크를 같은 차례로 되돌려주기 위해서다.
            for index, (kind, coords) in enumerate(self._prompts):
                if kind == "point":
                    x, y = coords
                    predictor.add_new_points_or_box(
                        state,
                        frame_idx=0,
                        obj_id=index + 1,
                        points=np.array([[float(x), float(y)]], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32),
                    )
                elif kind == "box":
                    predictor.add_new_points_or_box(
                        state,
                        frame_idx=0,
                        obj_id=index + 1,
                        box=np.array([float(value) for value in coords]),
                    )
                else:
                    raise ValueError(f"모르는 프롬프트 종류입니다: {kind}")
        self._predictor_ref = predictor
        self._state = state

    def track(self, bgr_image):
        """현재 프레임에서 티칭한 대상들의 마스크.

        Returns:
            prompts 와 같은 차례의 목록. 못 찾은 자리는 None.
        """
        with self._lock:
            self._ensure_ready()
            predictor = self._predictor_ref
            state = self._state
            found = [None] * len(self._prompts)
            with torch.inference_mode(), _inference_context():
                # CPU 텐서를 GPU 슬라이스에 넣으면 대입이 알아서 옮겨 준다.
                state["images"][1] = _to_tensor(bgr_image, predictor.image_size)
                # frame 1 의 인코딩 결과와 지난 촬영의 추적 결과를 지운다.
                # 남겨 두면 예측기가 이번 프레임을 새로 보지 않는다.
                state["cached_features"] = {}
                for outputs in state["output_dict_per_obj"].values():
                    outputs["non_cond_frame_outputs"].pop(1, None)
                for tracked in state["frames_tracked_per_obj"].values():
                    tracked.pop(1, None)

                for frame_idx, obj_ids, masks in predictor.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=1
                ):
                    # frame 0 은 조건부 프레임이라 다시 계산하지 않고 지나간다.
                    if frame_idx != 1:
                        continue
                    # obj_ids 는 넣은 차례대로 온다. 그래도 값으로 자리를 되찾아
                    # 두는 편이, 나중에 대상을 지우는 기능이 붙어도 어긋나지 않는다.
                    for position, obj_id in enumerate(obj_ids):
                        index = int(obj_id) - 1
                        if 0 <= index < len(found):
                            found[index] = (masks[position, 0] > 0.0).cpu().numpy()
        return found


# 상태 하나가 1024x1024 프레임 두 장과 그 특징을 들고 있어 제법 무겁다.
# 다만 홀용과 법선 매칭점용을 동시에 써야 하므로 하나만 남겨서는 안 된다
# (하나만 두면 촬영마다 서로를 밀어내 티칭 프레임을 다시 인코딩하게 된다).
_MAX_CACHED_SCENES = 4
_scenes = OrderedDict()
_scenes_lock = threading.Lock()


def _scene_key(scene_bgr, prompts, model_name):
    """같은 티칭인지 값싸게 가리는 열쇠 (sam2_detector._image_key 와 같은 방식)."""
    sample = scene_bgr[::37, ::41]
    return (
        model_name,
        scene_bgr.shape,
        int(sample.sum()),
        tuple(
            (kind, tuple(int(value) for value in coords))
            for kind, coords in prompts
        ),
    )


def _taught_scene(model_name, scene_bgr, prompts):
    key = _scene_key(scene_bgr, prompts, model_name)
    with _scenes_lock:
        scene = _scenes.get(key)
        if scene is None:
            scene = _TaughtScene(model_name, scene_bgr, prompts)
            _scenes[key] = scene
            while len(_scenes) > _MAX_CACHED_SCENES:
                _scenes.popitem(last=False)   # 가장 오래된 것부터 버린다
        else:
            _scenes.move_to_end(key)
        return scene


# ======================================================
# 재검출 (오퍼레이터 / 재분석)
# ======================================================
class Sam2VideoDetector:
    """티칭해 둔 홀을 새 이미지에서 추적한다.

    BlobAnalyzer / TemplateMatcher 와 같은 자리에 끼우려고
    analyze() 의 모양을 똑같이 맞췄다. 그래야 이 아래의 면 클러스터링·법선·포즈
    계산이 검출 방식을 몰라도 그대로 돈다.

    roi 는 받기만 하고 쓰지 않는다. 어디를 볼지 정하는 것이 이 방식의 요점인데
    미리 자른 범위로 묶으면 거리·회전에 강인해진 이유가 사라지기 때문이다.
    """

    def __init__(self, scene_bgr, box, model_name=None, taught_area=None):
        """
        Args:
            scene_bgr: 티칭할 때 본 화면 전체 (크롭이 아니다)
            box: 티칭 때 SAM2 가 잡은 홀 경계의 외접 사각형
                (left, top, right, bottom). 260908 에 점 프롬프트에서 바뀌었다 —
                법선 매칭점과 같은 형태로 맞춰 저장 구조와 재검출 경로를 하나로
                쓰기 위해서다. 추적 성능은 실측에서 점과 같았다(놓침 0회, 반지름
                변동 1.7%로 동일).
            model_name: 쓸 체크포인트. None 이면 기본값.
            taught_area: 티칭 때 비디오 추적으로 잰 마스크 넓이. 재검출이
                '제대로 잡혔나'를 이 값에 견준다(_min_area_for). None 이면
                절대 하한만 쓴다.
        """
        if scene_bgr is None or getattr(scene_bgr, "size", 0) == 0:
            raise ValueError("티칭 프레임이 없습니다.")
        box = tuple(box) if box is not None else ()
        if len(box) != 4:
            raise ValueError("티칭할 때 잡은 홀 네모가 없습니다. 홀을 다시 지정하세요.")
        # info.json 을 손으로 고쳤을 수도 있다. 장면 밖의 네모를 그대로 넘기면
        # SAM2 가 배경을 물어 엉뚱한 좌표가 나오므로 여기서 막는다.
        scene_height, scene_width = scene_bgr.shape[:2]
        left, top, right, bottom = box
        if not (0 <= left < right <= scene_width and 0 <= top < bottom <= scene_height):
            raise ValueError(
                f"티칭할 때 잡은 홀 네모 {box} 가 티칭 프레임 "
                f"({scene_width}x{scene_height}) 밖입니다. 홀을 다시 지정하세요."
            )
        self._scene = scene_bgr
        self._box = tuple(int(value) for value in box)
        self._model_name = model_name
        self._taught_area = int(taught_area or 0)

    def analyze(self, bgr_image, roi=None):
        """새 이미지에서 티칭한 홀을 추적한다.

        Returns:
            (holes, debug_info) — BlobAnalyzer 와 같은 모양.
            holes 는 [(x, y, radius)] 이고 못 찾으면 빈 목록이다.
            debug_info["binary"] 는 홀 마스크(원본 크기 uint8).
            debug_info["hole_masks"] 는 홀마다의 bool 마스크로, 화면에
            컨투어를 그리는 데 쓴다. 원과 중심만으로는 SAM2 가 실제로 무엇을
            물었는지 알 수 없어, 빗나갔을 때 알아채지 못한다.
        """
        height, width = bgr_image.shape[:2]
        binary = np.zeros((height, width), dtype=np.uint8)
        empty = {"binary": binary, "hole_masks": []}

        mask = _track_one(
            self._model_name, self._scene, ("box", self._box), bgr_image, "홀"
        )
        if mask is None:
            return [], dict(empty, sam2_stage="track_miss")
        area = int(mask.sum())
        if area > height * width * _MAX_AREA_RATIO:
            # 홀이 아니라 판재를 물었다. 추적이 어긋난 것이다.
            return [], dict(empty, sam2_stage="track_too_large")
        # 아래로도 막는다. 260908 이전에는 상한만 봐서, 마스크가 쪼그라들어도
        # 그대로 통과하고 반지름만 조용히 작아졌다. 매칭점과 같은 상대 기준을
        # 쓴다 — 절대 픽셀 수로 재면 홀 크기가 다른 지점에서 뜻이 달라진다.
        if area < _min_area_for(self._taught_area):
            return [], dict(empty, sam2_stage="track_too_small")

        hole = mask_to_hole(mask)
        if hole is None:
            return [], dict(empty, sam2_stage="track_miss")

        binary[mask] = 255
        return [hole], {"binary": binary, "hole_masks": [mask], "sam2_stage": "ok"}


def _check_resolution(scene, bgr_image, what):
    """티칭 프레임과 촬영 해상도가 같은지. 다르면 사유와 함께 예외."""
    height, width = bgr_image.shape[:2]
    if scene.scene_shape == (height, width):
        return
    taught_h, taught_w = scene.scene_shape
    raise RuntimeError(
        f"티칭할 때와 촬영 해상도가 다릅니다 "
        f"(티칭 {taught_w}x{taught_h}, 지금 {width}x{height}). "
        f"카메라 해상도를 맞추거나 {what}을 다시 지정하세요."
    )


def _track_one(model_name, scene_bgr, prompt, bgr_image, what):
    """대상 하나를 추적한다. 마스크 또는 None."""
    scene = _taught_scene(model_name, scene_bgr, (prompt,))
    _check_resolution(scene, bgr_image, what)
    return scene.track(bgr_image)[0]


class Sam2VideoNormalPoints:
    """법선을 세울 매칭점 **세 곳**을 새 이미지에서 함께 추적한다.

    왜 면 하나가 아니라 점 셋인가
        면 하나를 통째로 잡아 그 뎁스로 평면을 맞추면, 부품이 놓인 자리가
        틀어질 때 면 안에서 표본이 쓸리는 만큼 법선이 흔들린다. 대신 부품
        위의 **같은 자리 세 곳**을 각각 추적해 그 중심점으로 평면을 세우면,
        부품이 움직여도 세 점이 함께 따라가므로 평면이 부품에 붙는다.

    왜 셋에서 늘리지 않는가
        NORMAL_POINTS 설명 참고. 넷 이상은 '모두 같은 평면에 있다'를 전제하는
        근사가 되어, 한 점이 다른 단차에 올라가 있어도 알아챌 수 없게 된다.

    왜 한 상태에 함께 넣는가
        점마다 따로 추적기를 돌리면 촬영마다 티칭 프레임 인코딩(약 1.5초)이
        세 번 되풀이된다. 세 점은 모두 같은 티칭 프레임에 있으므로 한 상태에
        객체로 넣으면 인코딩은 한 번이고, 점마다 마스크 디코딩만 더 든다.

    평면을 세우는 일 자체는 여기서 하지 않는다. 여기는 '어디인지'까지만 알려
    주고, 그 자리의 뎁스를 읽어 평면을 맞추는 것은 부르는 쪽 몫이다 —
    뎁스와 카메라 파라미터를 아는 쪽이 거기이기 때문이다
    (normal_plane.plane_from_normal_points).
    """

    def __init__(self, scene_bgr, boxes, model_name=None, taught_areas=None):
        """
        Args:
            scene_bgr: 티칭할 때 본 화면 전체
            boxes: [(left, top, right, bottom)] 정확히 세 개.
            model_name: 쓸 체크포인트. None 이면 기본값.
            taught_areas: 티칭 때 **비디오 추적으로 잰** 넓이 세 개
                (measure_normal_point_areas). 통과 문턱을 이 값에 견준다.
                None 이면 절대 하한만 쓴다(옛 자산).
        """
        if scene_bgr is None or getattr(scene_bgr, "size", 0) == 0:
            raise ValueError("티칭 프레임이 없습니다.")
        boxes = [tuple(box) for box in (boxes or [])]
        if len(boxes) != NORMAL_POINTS:
            raise ValueError(
                f"매칭점은 정확히 {NORMAL_POINTS}개여야 합니다 (지금 {len(boxes)}개)."
            )
        scene_height, scene_width = scene_bgr.shape[:2]
        for index, (left, top, right, bottom) in enumerate(boxes, start=1):
            if not (
                0 <= left < right <= scene_width
                and 0 <= top < bottom <= scene_height
            ):
                raise ValueError(
                    f"{index}번 매칭점의 네모가 티칭 프레임 "
                    f"({scene_width}x{scene_height}) 밖입니다. 다시 지정하세요."
                )
        self._scene = scene_bgr
        self._boxes = boxes
        self._model_name = model_name
        # 개수가 안 맞으면 아예 안 쓴다. 자리가 밀린 기준으로 재면 엉뚱한 점이
        # 탈락하는데, 그건 기준이 없는 것보다 나쁘다.
        if taught_areas is not None and len(taught_areas) == len(boxes):
            self._taught_areas = [int(area or 0) for area in taught_areas]
        else:
            self._taught_areas = [0] * len(boxes)

    def __len__(self):
        return len(self._boxes)

    def find_masks(self, bgr_image):
        """새 이미지에서 매칭점 세 곳을 찾는다.

        Returns:
            ([마스크 또는 None] 세 개, 무슨 일이 있었는지 알려 주는 문구)

            **하나만 놓쳐도 평면은 못 세운다** — 여분이 없다. 그래도 예외를
            올리지 않고 문구로 알린다. 법선을 못 구했다고 좌표를 아예 안 내는
            것보다, 홀 고리 법선으로 물러서고 그 사실을 남기는 편이 낫다
            (extract_mask 가 같은 판단을 한다).
        """
        height, width = bgr_image.shape[:2]
        prompts = tuple(("box", box) for box in self._boxes)
        scene = _taught_scene(self._model_name, self._scene, prompts)
        _check_resolution(scene, bgr_image, "매칭점")

        limit = height * width * MAX_REGION_AREA_RATIO
        masks = []
        missed = []
        for index, mask in enumerate(scene.track(bgr_image), start=1):
            taught = self._taught_areas[index - 1]
            floor = _min_area_for(taught)
            area = 0 if mask is None else int(mask.sum())
            # 넓이 0 은 '작다'가 아니라 **그 자리를 못 찾았다**는 뜻이다. 둘을
            # 같은 말로 묶으면 화면 밖으로 나간 것(추적 실패)과 문턱에 걸린 것
            # (판정 문제)을 가릴 수 없어, 무엇을 손봐야 할지 알 수 없다.
            if area == 0:
                masks.append(None)
                missed.append(f"{index}번(못 찾음)")
                continue
            # 너무 작으면 잡티고, 너무 크면 부품 전체를 문 것이다. 어느 쪽이든
            # 그 자리의 뎁스를 대표하지 못하므로 평면 계산에 넣지 않는다.
            #
            # 작은 쪽 기준은 **티칭 때 넓이에 견준다**(NORMAL_POINT_MIN_AREA_RATIO).
            # 왜 사유를 적어 두는가: 예전에는 그냥 '놓쳤습니다' 뿐이라 추적이
            # 빗나간 것인지 문턱에 걸린 것인지 가릴 수 없었다. 실제로 16px 차이로
            # 탈락하던 것을 아무도 못 짚었다.
            if area < floor:
                masks.append(None)
                missed.append(f"{index}번(작음 {area}<{floor})")
                continue
            if area > limit:
                masks.append(None)
                missed.append(f"{index}번(너무 큼 {area})")
                continue
            masks.append(mask)

        if not missed:
            return masks, ""
        return masks, (
            f"매칭점 {', '.join(missed)}을 놓쳤습니다 "
            f"— 세 곳이 다 있어야 평면을 세울 수 있습니다"
        )
