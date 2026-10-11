"""실행 형태(소스 실행 / exe 실행)에 따라 달라지는 기준 폴더를 한 곳에서 정한다.

PyInstaller로 묶으면 .py 모듈이 _internal 안의 압축 아카이브로 들어가므로
`os.path.dirname(__file__)` 이 실제로 존재하지 않는 경로를 가리킨다.
그래서 기준 폴더를 두 가지로 나눠 쓴다.

    소스 실행                          exe 실행 (onedir)
    Fastening_automation/             배포폴더/
      ├ PythonUI.py                     ├ Fasten Automation.exe
      ├ bolt.png        <- 리소스        ├ _internal/
      ├ Settings/       <- 사용자 데이터 │    └ bolt.png      <- 리소스
      ├ Calibration/    <- 사용자 데이터 ├ Settings/         <- 사용자 데이터
      └ Templates/      <- 사용자 데이터 ├ Calibration/      <- 사용자 데이터
                                        └ Templates/        <- 사용자 데이터

- resource_dir(): 함께 배포되는 읽기 전용 파일(아이콘 등). exe에서는 _internal.
- user_data_dir(): 프로그램이 쓰고 지우는 파일. exe에서는 exe가 놓인 폴더.
  _internal 안에 쓰면 재배포할 때 세팅이 날아가고, 폴더 권한에 막힐 수도 있다.

소스로 실행할 때는 둘 다 이 파일이 있는 폴더라서 기존 동작과 똑같다.
"""

import os
import sys

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))


def is_frozen():
    """PyInstaller로 묶인 exe로 실행 중인지."""
    return getattr(sys, "frozen", False)


def resource_dir():
    """함께 배포되는 읽기 전용 파일이 있는 폴더."""
    if is_frozen():
        # onedir 에서는 _internal, onefile 에서는 임시 해제 폴더를 가리킨다.
        return sys._MEIPASS
    return _SOURCE_DIR


def user_data_dir():
    """프로그램이 만들고 고치는 파일을 둘 폴더."""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return _SOURCE_DIR


def resource_path(*parts):
    return os.path.join(resource_dir(), *parts)


def user_data_path(*parts):
    return os.path.join(user_data_dir(), *parts)
