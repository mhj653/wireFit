"""오퍼레이터 모드 진입 페이지.

세팅의 마지막 페이지. 여기서 운영에 쓸 값을 다 정하고 넘어간다.

    공정명 [____]
    데이터 저장 [ ●]  [경로 선택] [.../Logs        ]
    통신 설정 [ ●]  IP [0.0.0.0 ]  Port [5000 ]  프로토콜 [▼]  회전 순서 [▼]

    체결 세팅
    모델 추가 [+] [−]
    ╱Model1╲╱Model2╲                       체결 포인트 추가 [+] [−]
    ┌──────┬───────────┬────────────────────────────────┐
    │ 순서 │ 비전 세팅 │ 캘리브레이션                   │
    └──────┴───────────┴────────────────────────────────┘

                                  [오퍼레이터 모드로 전환 ▶]

모델과 체결 포인트 (260910)
───────────────────────────
    모델        아예 다른 제품. 이름으로 구분하고 탭으로 넘나든다.
    체결 포인트  그 제품 **안에서** 체결할 자리. 모델의 하위 개념이다.

그래서 표는 **고른 모델의 것**이고, 모델을 바꾸면 내용이 통째로 갈린다.
모델 목록은 Settings/Models.json 에 **바꿀 때마다 바로** 저장한다(model_store).
공정명·데이터 저장·통신 설정은 모델과 무관한 전역 값이라 세션에 그대로 둔다.

MainWindow가 _load_pages에서 bind_window()로 자신을 주입하면,
전환 버튼이 window.enter_operator_mode(...)를 호출한다.

콤보에 채울 목록(비전 세팅 / 캘리브레이션)은 이 페이지가 화면에 나올 때마다
다시 읽는다. 앞 탭에서 세팅을 새로 만들거나 캘리브레이션을 새로 뜨고 와도
목록에 바로 보여야 하기 때문이다.
"""

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

import calibration_store
import model_store
from fastening_table import FasteningTable
from operator_control_panel import OperatorControlPanel
from operator_panels import ModelTabBar
from pages.base_page import BasePage
from settings_store import list_profiles
from tcp_settings_panel import TcpSettingsPanel
from text_input_dialog import TextInputDialog
from theme import CONTENT_SPACING


class OperatorLaunchPage(BasePage):
    """공정명·저장 설정·모델별 체결 포인트를 정하고 오퍼레이터 모드로 넘어가는 페이지."""

    title = "Operator Mode"
    nav_icon = "operator"
    # 표가 창 높이에 맞춰 늘어나야 하므로 콘텐츠가 남는 공간을 다 쓰게 한다.
    expand_content = True

    def __init__(self, parent=None):
        super().__init__(parent)
        self._window = None
        # 모델 목록 [{"name": str, "fastening_points": [dict, ...]}, ...]
        self._models = []
        self._active_index = -1
        self._switching = False   # 모델을 갈아 끼우는 중 (저장 되먹임 방지)
        # 저장된 모델 파일이 없어서 기본 모델로 시작했는지. 그때만 예전 세션의
        # 체결 포인트를 옮겨 온다(restore_session_state).
        self._needs_migration = False

        self.content_layout.addWidget(
            QLabel(
                "공정명과 저장 경로, 모델별 체결 포인트를 정하고 "
                "오퍼레이터 모드로 전환하세요."
            )
        )

        self.control_panel = OperatorControlPanel()
        self.content_layout.addWidget(self.control_panel)

        self.tcp_panel = TcpSettingsPanel()
        self.content_layout.addWidget(self.tcp_panel)

        # 표만 stretch 를 받는다. 그래서 창이 커지면 표가 늘어나고
        # 위의 설정들과 전환 버튼은 제자리에 있는다.
        self.content_layout.addLayout(self._build_model_row())
        self.content_layout.addWidget(self._build_table(), 1)

        # 전환 버튼은 오른쪽 아래에 둔다(설정을 위에서 아래로 훑고 마지막에 누름).
        button_row = QHBoxLayout()
        button_row.addStretch()
        self.btn_switch = QPushButton("오퍼레이터 모드로 전환  ▶")
        self.btn_switch.setObjectName("primaryBtn")
        self.btn_switch.clicked.connect(self._switch)
        button_row.addWidget(self.btn_switch)
        self.content_layout.addLayout(button_row)

        self._load_models()
        self.refresh_options()

    def bind_window(self, window):
        self._window = window

    # ===== UI 구성 =====
    def _build_model_row(self):
        """'체결 세팅' 제목 + 모델 추가 [+] [−]."""
        column = QVBoxLayout()
        column.setSpacing(CONTENT_SPACING // 2)

        caption = QLabel("체결 세팅")
        caption.setObjectName("sectionLabel")
        column.addWidget(caption)

        row = QHBoxLayout()
        row.setSpacing(CONTENT_SPACING // 2)

        label = QLabel("모델 추가")
        label.setObjectName("paramLabel")
        row.addWidget(label)

        self.btn_add_model = QPushButton("+")
        self.btn_add_model.setObjectName("iconBtn")
        self.btn_add_model.setToolTip("새 제품 모델을 만든다 (이름을 입력한다)")
        self.btn_add_model.clicked.connect(self._on_add_model)
        row.addWidget(self.btn_add_model)

        self.btn_remove_model = QPushButton("−")
        self.btn_remove_model.setObjectName("iconBtn")
        self.btn_remove_model.setToolTip(
            "고른 모델을 삭제한다 (체결 포인트까지 사라진다)"
        )
        self.btn_remove_model.clicked.connect(self._on_remove_model)
        row.addWidget(self.btn_remove_model)

        row.addStretch()
        column.addLayout(row)
        return column

    def _build_table(self):
        """모델 탭 + 체결 포인트 표.

        탭은 표 위 줄의 왼쪽 자리에 넣는다(FasteningTable.set_leading_widget).
        표 바로 위에 붙어 있어야 '이 표가 그 모델의 것'으로 읽힌다.
        """
        self.table = FasteningTable()
        self.model_tabs = ModelTabBar()
        self.model_tabs.model_selected.connect(self._on_model_tab_selected)
        self.table.set_leading_widget(self.model_tabs)
        # 체결 포인트를 고치면 그 자리에서 모델 파일에 쓴다.
        self.table.data_changed.connect(self._save_models)
        return self.table

    def showEvent(self, event):
        """탭이 열릴 때마다 콤보 목록을 다시 읽는다."""
        super().showEvent(event)
        self.refresh_options()

    def refresh_options(self):
        self.table.set_blob_options(list_profiles())
        self.table.set_calibration_options(calibration_store.list_sessions())

    # ===== 모델 =====
    def _load_models(self):
        """저장된 모델 목록을 읽어 탭과 표에 싣는다.

        파일이 없으면(첫 실행) 기본 모델 하나로 시작한다. 모델이 하나도 없으면
        체결 포인트를 담을 자리가 없어 표를 띄울 수 없다.
        """
        models, active = model_store.load_models()
        if not models:
            models = [{"name": model_store.DEFAULT_MODEL_NAME,
                       "fastening_points": []}]
            active = models[0]["name"]
            # 예전 세션에 체결 포인트가 남아 있으면 이 모델로 옮겨 온다.
            self._needs_migration = True
        self._models = models
        names = [entry["name"] for entry in models]
        self._active_index = names.index(active) if active in names else 0
        self._reload_model_tabs()

    def _reload_model_tabs(self):
        """탭을 다시 짜고 고른 모델의 체결 포인트를 표에 싣는다."""
        self._switching = True
        try:
            self.model_tabs.set_models(
                [entry["name"] for entry in self._models],
                selected=max(0, self._active_index),
            )
            self._load_active_points()
        finally:
            self._switching = False
        # 마지막 하나까지 지우면 담을 자리가 없어지므로 그때는 삭제를 막는다.
        self.btn_remove_model.setEnabled(len(self._models) > 1)

    def _load_active_points(self):
        """고른 모델의 체결 포인트를 표에 싣는다(저장 알림 없이)."""
        entry = self._active_model()
        points = entry.get("fastening_points") if entry else []
        self.table.set_data(points or [], notify=False)

    def _active_model(self):
        if 0 <= self._active_index < len(self._models):
            return self._models[self._active_index]
        return None

    def _commit_active_points(self):
        """표에 있는 값을 고른 모델에 담는다(모델을 옮기기 전에 부른다)."""
        entry = self._active_model()
        if entry is not None:
            entry["fastening_points"] = self.table.get_data()

    def _save_models(self):
        """모델 목록을 파일에 쓴다. 바뀌는 즉시 부른다.

        모델을 갈아 끼우는 중에는 쓰지 않는다 — 그때 표에 있는 값은 아직 어느
        모델의 것인지 정해지지 않았다.
        """
        if self._switching:
            return
        self._commit_active_points()
        entry = self._active_model()
        model_store.save_models(
            self._models, active=entry["name"] if entry else None
        )

    def current_model_name(self):
        entry = self._active_model()
        return None if entry is None else entry["name"]

    def model_entries(self):
        """모델 목록을 (이름, 체결 포인트 dict 목록) 그대로 돌려준다.

        오퍼레이터 화면의 '모델 변경' 창이 목록을 받아 간다. 표에 있는 값을
        먼저 담아 두므로, 저장을 누르지 않아도 방금 고친 값이 넘어간다.
        """
        self._commit_active_points()
        return [
            {"name": entry["name"],
             "fastening_points": list(entry.get("fastening_points") or [])}
            for entry in self._models
        ]

    def select_model(self, name):
        """이름으로 모델을 고른다. 오퍼레이터 화면에서 바꿨을 때 따라온다.

        260910 사용자 결정: 프로그램 전체에서 '현재 모델'은 하나다. 그래서
        오퍼레이터에서 바꾸면 이 화면의 탭도 같이 옮겨진다.
        """
        names = [entry["name"] for entry in self._models]
        if name not in names or names.index(name) == self._active_index:
            return False
        self._commit_active_points()
        self._active_index = names.index(name)
        self._reload_model_tabs()
        self._save_models()
        return True

    def _on_model_tab_selected(self, index):
        if self._switching or index == self._active_index:
            return
        # 넘어가기 전에 지금 표를 그 모델에 담아 둔다(안 담으면 방금 고친 값이
        # 날아간다).
        self._commit_active_points()
        self._active_index = index
        self._switching = True
        try:
            self._load_active_points()
        finally:
            self._switching = False
        self._save_models()

    def _on_add_model(self):
        name = TextInputDialog.get_text(
            self,
            "모델 추가",
            "새 제품 모델의 이름을 입력하세요. 체결 포인트는 모델마다 따로 관리됩니다.",
            ok_text="생성",
            cancel_text="취소",
            validator=lambda text: model_store.validate_model_name(
                text, [entry["name"] for entry in self._models]
            ),
        )
        if name is None:
            return   # 취소

        self._commit_active_points()
        # 새 모델은 빈 표로 시작한다. 다른 제품이므로 앞 모델의 체결 포인트를
        # 물려주면 안 된다 — 물려주면 지우는 것을 잊고 그대로 쓰게 된다.
        self._models.append({"name": name, "fastening_points": []})
        self._active_index = len(self._models) - 1
        self._reload_model_tabs()
        self._save_models()

    def _on_remove_model(self):
        entry = self._active_model()
        if entry is None or len(self._models) <= 1:
            return

        box = QMessageBox(self)
        box.setWindowTitle("모델 삭제")
        box.setIcon(QMessageBox.Warning)
        box.setText(f"'{entry['name']}' 모델을 삭제할까요?")
        box.setInformativeText(
            f"이 모델의 체결 포인트 "
            f"{len(entry.get('fastening_points') or [])}개가 함께 사라집니다. "
            "되돌릴 수 없습니다."
        )
        delete_button = box.addButton("삭제", QMessageBox.DestructiveRole)
        cancel_button = box.addButton("취소", QMessageBox.RejectRole)
        box.setDefaultButton(cancel_button)   # 실수로 엔터를 쳐도 지워지지 않게
        box.exec_()
        if box.clickedButton() is not delete_button:
            return

        self._models.pop(self._active_index)
        # 지운 자리에 오는 모델을 고른다(마지막이었다면 그 앞).
        self._active_index = min(self._active_index, len(self._models) - 1)
        self._reload_model_tabs()
        self._save_models()

    # ===== 재시작 상태 =====
    def session_state(self):
        """세션에 남길 값.

        체결 포인트는 여기 없다 — 모델 파일(Models.json)이 주인이다. 두 곳에
        두면 어느 쪽이 맞는지 알 수 없다.
        """
        return {
            "process_name": self.control_panel.get_process_text(),
            "save_enabled": self.control_panel.is_save_enabled(),
            "save_path": self.control_panel.get_save_path(),
            "server": {
                "enabled": self.tcp_panel.is_server_enabled(),
                "host": self.tcp_panel.get_host(),
                "port": self.tcp_panel.get_port(),
                # 프로토콜과 회전 순서는 로봇에 맞춰 한 번 고르면 계속 그대로
                # 쓰는 값이라, 다음 실행에서도 되살아나야 한다.
                "mode": self.tcp_panel.get_mode(),
                "rotation_order": self.tcp_panel.get_rotation_order(),
            },
        }

    def restore_session_state(self, state):
        if not isinstance(state, dict):
            return

        process_name = state.get("process_name")
        if isinstance(process_name, str):
            self.control_panel.set_process_name(process_name)

        self.control_panel.set_save_enabled(bool(state.get("save_enabled", False)))
        save_path = state.get("save_path")
        if isinstance(save_path, str) and save_path:
            self.control_panel.set_save_path(save_path)

        self._migrate_session_points(state.get("fastening_points"))

        server = state.get("server")
        if not isinstance(server, dict):
            return
        host = server.get("host")
        if isinstance(host, str):
            self.tcp_panel.set_host(host)
        port = server.get("port")
        if isinstance(port, int) and 1 <= port <= 65535:
            self.tcp_panel.set_port(port)
        mode = server.get("mode")
        if isinstance(mode, str):
            # 목록에 없는 값이면 패널이 무시하고 기본값을 지킨다.
            self.tcp_panel.set_mode(mode)
        rotation_order = server.get("rotation_order")
        if isinstance(rotation_order, str):
            self.tcp_panel.set_rotation_order(rotation_order)
        self.tcp_panel.set_server_enabled(bool(server.get("enabled", False)))

    def _migrate_session_points(self, points):
        """모델이 없던 시절의 체결 포인트를 기본 모델로 옮긴다.

        260910 이전 세션에는 체결 포인트가 세션 파일에 한 벌만 있었다. 모델
        파일이 없을 때(= 이번에 처음 올라온 것)만 옮긴다 — 모델 파일이 이미
        있으면 그쪽이 맞는 값이고, 세션의 옛 목록은 지나간 것이다.
        """
        if not self._needs_migration:
            return
        self._needs_migration = False
        if not isinstance(points, list):
            return
        points = [item for item in points if isinstance(item, dict)]
        if not points:
            return
        entry = self._active_model()
        if entry is None:
            return
        entry["fastening_points"] = points
        self._switching = True
        try:
            self._load_active_points()
        finally:
            self._switching = False
        self._save_models()

    def _switch(self):
        if self._window is None:
            return

        # 서버를 켠 채로 넘어가려면 IP/포트가 쓸 수 있는 값이어야 한다.
        # 여기서 막지 않으면 운영 화면에 들어간 뒤에야 안 붙는 걸 알게 된다.
        if self.tcp_panel.is_server_enabled():
            ok, reason = self.tcp_panel.is_valid()
            if not ok:
                QMessageBox.warning(self, "통신 설정", reason)
                return

        self._commit_active_points()
        self._save_models()
        self._window.enter_operator_mode(
            process_name=self.control_panel.get_process_name(),
            save_enabled=self.control_panel.is_save_enabled(),
            save_path=self.control_panel.get_save_path(),
            fastening_points=self.table.get_points(),
            server_enabled=self.tcp_panel.is_server_enabled(),
            server_config=self.tcp_panel.get_server_config(),
            models=self.model_entries(),
            active_model=self.current_model_name(),
        )
