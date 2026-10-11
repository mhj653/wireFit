# -*- mode: python ; coding: utf-8 -*-
r"""PyInstaller onedir build for the 260908 release.

Build:
    FastenAutomationVenv\Scripts\pyinstaller.exe --noconfirm --clean build_exe_260908.spec

    Run it with the sandbox disabled. Under the sandbox the build stops dead at
    "Looking for dynamic libraries": resolving the 4 GB of CUDA DLLs in torch/lib needs
    System32 reads and isolated child processes, and this spec itself copies the MSVC
    runtime out of System32. Expect roughly 34 minutes either way — the Analysis stage
    reruns in full even with a warm cache, so batch source edits before building.

    Close the application first. If FastenAutomation is running, COLLECT succeeds but
    _unify_msvc_runtime() below dies with [WinError 32]; the bundle is complete at that
    point, so rerun just that post-step instead of paying for a full rebuild.

Result:
    dist/FastenAutomation_260908/
      FastenAutomation_260908.exe
      _internal/
        models/sam2/*.pt        SAM2 checkpoints (~1.3 GB)
        torch/lib/*.dll         CUDA runtime (~4 GB)

Settings, Calibration, Templates, and Preset are user data, so copy their initial contents
beside the executable after PyInstaller finishes.

    Preset/ holds RealSense Advanced Mode presets saved from the RealSense Viewer. The
    Camera Selection page lists whatever .json files are in there, so shipping the folder
    empty just means the combo shows only "사용 안 함".

    Templates/ also holds SAM2 teaching data (scene.png + info.json). Hole references
    taught before 260907 have no scene.png and cannot be used with video tracking or the
    three-point normal, so they must be re-taught on the target machine.

What changed from 260907
    1. Pose Setting is now its own page. Hole Detection answers "where are the holes"
       and stops there; the three-point normal, the pose adjust offset, and the approach
       angle offset all moved to pages/pose_setting_page.py, which turns hole pixels into
       (x, y, z, rx, ry, rz).

       Two new modules exist because of that split. pose_pipeline.py holds the offset and
       normal maths both pages plus the operator path share, and image_board.py holds the
       image boards. They are separate modules because pose_setting_page imports
       hole_detection_page, so anything shared cannot live in either one without creating
       an import cycle. MainWindow._link_pages() hands the two pages to each other.

       The profile file is still one file per fastening point: Hole Detection owns it and
       merges the pose keys in at save time, because the operator's fastening table points
       at a profile by a single name.

    2. A matching point now passes or fails against the area it had when it was taught,
       not against a fixed pixel count. The old absolute floor of 200 px was rejecting
       points the tracker was finding perfectly well — a point taught at 204 px tracked at
       184 px and was dropped by 16 px on every single capture, which killed the whole
       plane because all three points are required.

       The taught area is measured with the video predictor at save time
       (measure_normal_point_areas) and stored in info.json as "normal_areas". Measuring it
       with the image predictor instead would be wrong: the two paths disagree badly, and
       one real point measured 3613 px one way and 174 px the other.

       Sets saved before this key exists still load; they fall back to the absolute floor,
       which is now 40 px and derives from the 10 depth samples point_from_mask needs.

    3. Missed matching points now say why — "작음 184<200" versus "못 찾음" — because the
       magenta box drawn at the taught coordinates looks identical whether the tracker lost
       the region, the region was filtered out, or the tracker never ran at all.

    4. Pose Setting picks where the orientation comes from, and the choice is saved per
       profile, so each operator board can use a different one:

         플랜지 고정   Vision supplies x, y, z only; the robot keeps its current flange
                       orientation. Like 2D vision except z is real. Most robust — no
                       plane is fitted, so it needs 10 depth samples instead of 30.
         홀 주변 영역  Fits a plane to an annulus around each hole and uses its normal.
                       The radius is given in mm because that is a property of the part,
                       so the same physical area is used at any camera distance. The
                       yellow ring drawn on screen is the actual sample region — the
                       drawing and the sampling call the same ring_outer_px().
         매칭점 3곳    The previous behaviour, and still the most repeatable (0.11°).

       Profiles saved before this key exists read as 매칭점 3곳, which is what they used.

    5. Per-capture EMA smoothing of the normal is gone. It blended 80% of the previous
       capture and moved at most 3° per shot, so a part that actually moved took several
       captures to catch up and the result looked like it was tracking the previous frame.
       stabilize_hole_poses() and the previous_poses plumbing are still there, one line
       from being re-enabled.

    6. Pose Setting re-reads Hole Detection's capture when a newer one exists. It used to
       ingest only once, so moving the part and re-capturing left the old position on
       screen — which read as the vision not responding to position changes at all.

    7. The first-run default profile now contains the pose keys. init_profiles() moved out
       of HoleDetectionPage.__init__ to after MainWindow._link_pages(), because before the
       two pages are joined there is nowhere to ask for those values.

    torch, torchvision, sam2, and hydra all have hooks in pyinstaller-hooks-contrib, so
    their config files, dynamic submodules, and the source .py files that torch.jit.script
    needs are collected without extra help here. The sam2 hook does collect_submodules,
    which is what pulls in sam2.sam2_video_predictor (referenced only as a hydra _target_
    string, so static analysis would miss it).

    torch stays the CUDA build (2.11.0+cu128). cu128 is the only index covering both target
    GPUs: RTX 4050 is sm_89 and RTX 5060 is sm_120 (Blackwell), and sm_120 kernels exist
    only in CUDA 12.8+ wheels. sm_89 is not in get_arch_list() for this wheel, yet the 4050
    works because cubins are binary compatible within a major version, which is why
    sam2_detector._pick_device() probes a real kernel instead of string-matching.
"""

import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# SAM2 체크포인트. 폴더째 datas 로 넣어 _internal/models/sam2/ 에 그대로 앉힌다.
# 파일 목록을 손으로 적지 않는 이유는, 나중에 체크포인트를 더 받아 두었을 때
# spec 을 고치는 것을 잊어 조용히 빠지는 일을 막기 위해서다.
# MSVC 런타임. torch 의 c10.dll 은 msvcp140.dll 을, torch_cpu.dll 은 추가로
# vcruntime140_threads.dll 을 요구하는데 PyInstaller 가 이 둘을 안 가져온다.
# 개발 PC 는 System32 에 있어 그냥 되지만, VC++ 재배포 패키지가 없는 PC 에서는
# "[WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다" 로 torch import 가 죽고
# SAM2 검출이 통째로 안 된다.
#
# PyQt5 가 Qt5/bin 에 msvcp140.dll 사본을 갖고 있지만 그것으로는 안 된다.
# 그 폴더는 PyQt5 가 올라온 뒤에야 검색 경로에 붙는데, 우리는 torch 를 PyQt5
# 보다 먼저 올리기 때문이다(PythonUI.py 의 선로드 참고).
#
# _internal 과 torch/lib 양쪽에 둔다. 앞은 PyInstaller 가, 뒤는 torch 가 직접
# os.add_dll_directory() 로 검색 경로에 넣는 자리다. 합쳐 1MB 남짓이라 어느
# 쪽이 먼저 잡히든 되게 해 두는 편이 낫다.
_SYSTEM32 = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')
_MSVC_RUNTIME = ['msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140_threads.dll']

runtime_binaries = []
for _name in _MSVC_RUNTIME:
    _src = os.path.join(_SYSTEM32, _name)
    if not os.path.isfile(_src):
        raise SystemExit(
            f"{_src} 가 없습니다. 이 PC 에 Visual C++ 재배포 패키지를 설치한 뒤 "
            "다시 빌드하세요. 이 DLL 없이 만들면 재배포 패키지가 없는 PC 에서 "
            "torch 가 뜨지 않습니다."
        )
    runtime_binaries.append((_src, '.'))
    runtime_binaries.append((_src, os.path.join('torch', 'lib')))

# torchvision 0.26 exposes torchvision._C, which the standard hook does find.
# Collecting explicitly anyway: it is cheap, and 0.29 (used in 260902) loaded
# _C_stable.pyd indirectly via torch.ops.load_library() where the hook missed it.
# Keeping this means a torchvision bump cannot silently drop the extensions.
torchvision_binaries = collect_dynamic_libs(
    'torchvision', search_patterns=['*.pyd', '*.dll']
)

_MODEL_DIR = os.path.join('models', 'sam2')
model_datas = [
    (os.path.join(_MODEL_DIR, name), _MODEL_DIR)
    for name in sorted(os.listdir(_MODEL_DIR))
    if name.endswith('.pt')
]
if not model_datas:
    raise SystemExit(
        f"{_MODEL_DIR} 에 SAM2 체크포인트(.pt)가 없습니다. "
        "빌드해도 SAM2 검출을 쓸 수 없습니다."
    )

a = Analysis(
    ['PythonUI.py'],
    pathex=[],
    binaries=runtime_binaries + torchvision_binaries,
    datas=[
        ('bolt.png', '.'),
        *model_datas,
        # sam2 훅이 이미 챙기지만, 훅이 없는 환경에서도 config 가 빠지지 않도록
        # 한 번 더 명시한다. 중복은 PyInstaller 가 알아서 정리한다.
        *collect_data_files('sam2'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'pytest',
        'IPython',
        'notebook',
        'numpy.testing',
        'PyQt5.QtWebEngineWidgets',
        'PyQt5.QtBluetooth',
        'PyQt5.QtQuick',
        'PyQt5.QtQml',
        # torch 의 하위 모듈은 빼지 말 것. 추론만 쓰더라도 torch/__init__.py 가
        # distributed 와 testing 을 무조건 import 하므로(2703, 2720행), 빼면
        # "No module named 'torch.distributed'" 로 import 자체가 실패한다.
        # 크기를 줄이려고 손댔다가 배포본이 통째로 안 뜬 적이 있다.
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FastenAutomation_260908',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='bolt.ico',
    manifest='windows_per_monitor_v2.manifest',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='FastenAutomation_260908',
)


# ======================================================
# COLLECT 뒤처리 — MSVC 런타임 사본 버전 통일
# ======================================================
# 번들 안에 msvcp140.dll 사본이 여러 개 들어온다. PyQt5 가 Qt5/bin 에 자기
# 것(14.26)을 갖고 있고, PyInstaller 가 파이썬용(14.38)을 _internal 에 넣는다.
#
# Windows 는 같은 이름의 DLL 을 경로가 달라도 프로세스에 하나만 올린다. 그래서
# 먼저 잡힌 사본이 이긴다. 실제로 PyQt5 의 14.26 이 로드되고, torch 의 c10.dll
# 은 훨씬 최신 MSVC 로 빌드돼 14.26 에 없는 export 를 요구하므로 초기화가
# 실패한다 -> "[WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다".
#
# 소스로 실행하면 System32 의 최신 런타임이 잡혀서 이 문제가 안 보인다. exe 로
# 묶었을 때만 드러나므로, 여기서 모든 사본을 System32 의 것으로 맞춰 둔다.
# MSVC 런타임은 상위 호환이라 Qt(14.26 기준 빌드)도 14.44 에서 그대로 돈다.
#
# 이름에 '-' 가 든 사본(numpy 의 msvcp140-<해시>.dll)은 건드리지 않는다.
# 그건 numpy 전용으로 이름을 바꿔 둔 것이라 이 충돌과 무관하다.
def _unify_msvc_runtime(dist_root):
    import shutil

    patterns = ('msvcp140', 'vcruntime140', 'concrt140')
    replaced = []
    for folder, _dirs, files in os.walk(dist_root):
        for name in files:
            low = name.lower()
            if not low.endswith('.dll') or '-' in low:
                continue
            if not low.startswith(patterns):
                continue
            source = os.path.join(_SYSTEM32, low)
            if not os.path.isfile(source):
                continue
            target = os.path.join(folder, name)
            shutil.copy2(source, target)
            replaced.append(os.path.relpath(target, dist_root))
    return replaced


_dist_root = os.path.join(DISTPATH, 'FastenAutomation_260908')
_replaced = _unify_msvc_runtime(_dist_root)
print(f"[spec] MSVC 런타임 사본 {len(_replaced)}개를 System32 버전으로 통일했습니다:")
for _item in _replaced:
    print(f"[spec]   {_item}")
if not _replaced:
    raise SystemExit(
        "MSVC 런타임 사본을 하나도 바꾸지 못했습니다. 이 상태로 배포하면 "
        "exe 에서 torch import 가 WinError 1114 로 실패합니다."
    )
