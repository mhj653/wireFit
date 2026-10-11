# Fastening Automation

로봇 체결 공정용 **비전 프로그램**. 로봇 손목에 단 RealSense 뎁스 카메라(D405)로
체결 홀을 찾고, 홀의 3D 위치·자세를 로봇 베이스 좌표 `(x, y, z, rx, ry, rz)`로
바꿔 TCP/IP로 로봇에 넘긴다.

PyQt5 데스크톱 앱(Windows). 화면은 두 가지다.

- **세팅 모드**: 카메라·홀 검출·자세·캘리브레이션을 맞추는 엔지니어용 화면
- **오퍼레이터 모드**: 로봇 신호를 받아 촬영·판정·좌표 전송만 하는 현장용 화면

## 주요 기능

세팅 모드의 사이드바 탭 순서가 곧 세팅 순서다.

| 탭 | 하는 일 |
|---|---|
| **Camera Selection** | RealSense 카메라 연결, 뎁스 프리셋·후처리 설정. 카메라 없이 저장된 이미지 폴더로 돌리는 **가상 카메라 모드** 지원 |
| **Hole Detection** | 홀 검출. 방식은 블랍 검출 / 템플릿 매칭 / SAM2 비디오 추적 중 선택. 체결 지점마다 프로파일로 저장 |
| **Pose Setting** | 홀 픽셀을 3D 자세로 변환. 자세 기준은 플랜지 고정 / 홀 주변 영역 / 매칭점 3곳 중 선택, 진입 각도 오프셋 티칭 |
| **EIH Calibration** | 체커보드로 Eye-in-Hand 캘리브레이션(카메라 ↔ 플랜지). **6축 / 4축 로봇** 선택 |
| **Operator Mode** | 모델(제품)·체결 포인트 구성, 저장 경로, TCP 서버 설정 후 오퍼레이터 화면으로 전환 |

오퍼레이터 화면 구성과 동작은 [OPERATOR_MODE.md](OPERATOR_MODE.md)에 정리돼 있다.

## 요구 사항

- Windows 10/11, **Python 3.13**
- Intel RealSense **D405** (없으면 가상 카메라 모드로 동작)
- NVIDIA GPU 권장 (SAM2 추적이 GPU에서 약 14배 빠르다). RTX 4050·5060에서 검증
- SAM2 체크포인트(.pt) — 용량이 커서 저장소에 없다. 아래 절차로 받는다

## 설치

```powershell
# 1) 가상환경 (exe 빌드 spec 이 이 이름을 쓴다)
python -m venv FastenAutomationVenv
FastenAutomationVenv\Scripts\activate

# 2) 의존성 (torch 는 requirements.txt 에 지정된 CUDA 12.8 빌드로 설치된다)
pip install -r requirements.txt
```

### SAM2 체크포인트 다운로드

`models/sam2/` 폴더에 내려받는다. SAM2 비디오 추적을 쓰지 않는다면 생략해도
되지만, exe 빌드에는 체크포인트가 하나 이상 있어야 한다.

```powershell
$dir = "models\sam2"; New-Item -ItemType Directory -Force -Path $dir | Out-Null
$base = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
foreach ($f in "sam2.1_hiera_tiny.pt","sam2.1_hiera_small.pt","sam2.1_hiera_base_plus.pt","sam2.1_hiera_large.pt") {
    Invoke-WebRequest "$base/$f" -OutFile (Join-Path $dir $f)
}
```

필요한 크기만 받아도 된다. 앱의 모델 콤보에서 고르며, 있는 것 중 가장 작은 모델이 기본값이다.

## 실행

```powershell
FastenAutomationVenv\Scripts\activate
python PythonUI.py
```

모듈을 상대 경로로 import 하므로 **이 폴더 안에서** 실행한다.

### 처음 실행하면 생기는 폴더

아래 폴더는 저장소에 없고, 프로그램이 처음 저장할 때 스스로 만든다
(exe로 실행하면 exe가 있는 폴더 아래에 생긴다).

| 폴더 | 내용 |
|---|---|
| `Settings/` | 홀 검출 프로파일, 모델(제품) 목록, 뎁스 후처리, 마지막 화면 상태 |
| `Calibration/` | EIH 캘리브레이션 결과 (`EIH_<날짜>_<시각>/` 마다 사진·포즈·result.json) |
| `Templates/` | 템플릿 매칭·SAM2 티칭 데이터 |
| `Preset/` | RealSense Viewer에서 저장한 D405 Advanced Mode 프리셋(.json). 여기에 넣으면 Camera Selection 콤보에 나온다 |
| `Logs/` | 오퍼레이터 모드 촬영 저장 (color / depth / depth_raw) |

`Logs/` 아래 날짜 폴더는 Camera Selection의 가상 카메라 폴더로 그대로 쓸 수 있다.

## TCP/IP 통신

로봇(클라이언트)이 서버(이 프로그램)에 쉼표로 구분한 아스키 한 줄을 보낸다.

| 명령 | 요청 예 | 성공 응답 | 실패 |
|---|---|---|---|
| 101 촬영 | `101,1,0,0` | `101,1102` | `101,1003` |
| 102 좌표 요청 | `102,1` | `102,1100,0,0,0,x,y,z,rx,ry,rz` | `102,1003` |
| 103 모델 변경 | `103,1,2` | `103,1107` | `103,1003` |

규약의 정본은 [protocol.py](protocol.py)의 `COMMANDS` 표다. 명령이나 항목을 바꿀 때는 그 표만 고친다.

## 테스트

```powershell
python -m unittest discover -p "test_*.py"
```

카메라·GPU 없이 돈다(합성 데이터 사용).

## exe 빌드

```powershell
FastenAutomationVenv\Scripts\pyinstaller.exe --noconfirm --clean build_exe_260908.spec
```

- 결과: `dist/FastenAutomation_260908/` (onedir). 약 34분 걸린다
- 빌드 전에 **실행 중인 앱을 닫는다** (MSVC 런타임 정리 단계가 파일 잠금으로 실패한다)
- 빌드 PC에 Visual C++ 재배포 패키지가 있어야 한다
- 자세한 주의사항은 spec 파일 맨 위 설명에 있다

## 폴더 구조

```
PythonUI.py              진입점 · 메인 창 (세팅/오퍼레이터 전환, 페이지 등록)
pages/                   세팅 모드 탭 (BasePage 하위 클래스 하나 = 탭 하나)
operator_view.py         오퍼레이터 화면
operator_*.py            오퍼레이터 화면 구성 요소
tcp_server.py            TCP 서버
protocol.py              통신 패킷 규약
hand_eye.py              EIH 캘리브레이션 계산 (6축 / 4축)
calibration_store.py     캘리브레이션 저장·로드
hole_pose.py             카메라 기준 홀 → 플랜지/베이스 좌표 변환
pose_pipeline.py         자세 계산 공용 로직 (Hole Detection · Pose Setting · 오퍼레이터)
normal_plane.py          홀 주변 면 법선
template_matching.py     템플릿 매칭
sam2_detector.py         SAM2 이미지 예측
sam2_video_tracker.py    SAM2 비디오 추적
realsense_*.py           RealSense 프리셋 · 뎁스 후처리
*_store.py               설정·모델·템플릿·세션 파일 저장
theme.py                 스타일시트 · 크기 상수
app_paths.py             소스 실행 / exe 실행의 기준 폴더
build_exe_260908.spec    PyInstaller 빌드 설정
test_*.py                단위 테스트
```

새 세팅 탭을 추가하려면 `pages/`에 `BasePage`를 상속한 클래스를 만들고
`PythonUI.py`의 `PAGES` 목록에 넣는다.

## 라이선스 / 사용한 오픈소스

- SAM 2 (Meta): Apache License 2.0 — https://github.com/facebookresearch/sam2
