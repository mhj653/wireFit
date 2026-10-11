"""체결 위치 목록.

체결 위치 하나가 표의 한 행이고, 각 행은 어떤 블롭 세팅과 어떤 캘리브레이션을
쓸지 고른 것이다.

순서 번호는 따로 담아 두지 않고 목록에서의 자리로 정한다. 드래그로 행을 옮겨도
번호 칸은 늘 위에서부터 1, 2, 3... 으로 읽히고, 옮겨 다니는 건 행의 내용뿐이다.
번호를 값으로 갖고 있으면 옮길 때마다 다시 매기는 코드가 필요하고, 한 군데를
놓치면 표에 보이는 번호와 저장된 번호가 어긋난다.
"""


class FasteningPoint:
    """체결 위치 하나."""

    def __init__(self, blob_setting="", calibration=""):
        self.blob_setting = blob_setting
        self.calibration = calibration

    def to_dict(self, order=None):
        data = {
            "blob_setting": self.blob_setting,
            "calibration": self.calibration,
        }
        if order is not None:
            data["order"] = order
        return data

    @staticmethod
    def from_dict(data):
        return FasteningPoint(
            blob_setting=data.get("blob_setting", ""),
            calibration=data.get("calibration", ""),
        )


class FasteningDataModel:
    """체결 위치 목록 + 변경 알림.

    알림이 두 갈래다.

      on_changed       표를 **다시 그려야** 하는 변경 (행 추가/삭제/이동/교체)
      on_data_changed  값만 바뀐 변경까지 모두 (위의 것 + 콤보 선택)

    나눠 둔 이유: 콤보에서 고른 값을 담을 때(update_point) 표를 다시 그리면,
    아직 신호를 처리하는 중인 그 콤보를 지우고 새로 만드는 일이 된다. 그래서
    다시 그리는 알림에는 태우지 못한다. 그런데 '값이 바뀌었으니 파일에 써 두라'
    는 쪽은 그때도 알아야 한다(모델 목록을 바꿀 때마다 저장한다).
    """

    def __init__(self):
        self._points = []
        self._listeners = []
        self._data_listeners = []

    # ===== 조회 =====
    def count(self):
        return len(self._points)

    def get_points(self):
        return list(self._points)

    def get_point(self, index):
        if 0 <= index < len(self._points):
            return self._points[index]
        return None

    def to_list(self):
        """저장용. 순서 번호는 여기서 자리대로 매긴다."""
        return [
            point.to_dict(order=index)
            for index, point in enumerate(self._points, start=1)
        ]

    # ===== 변경 =====
    def add_point(self):
        point = FasteningPoint()
        self._points.append(point)
        self._notify()
        return point

    def remove_point(self, index):
        if not (0 <= index < len(self._points)):
            return False
        self._points.pop(index)
        self._notify()
        return True

    def move_point(self, from_index, to_index):
        """행을 to_index 자리로 옮긴다.

        번호 칸은 자리로 정하므로 여기서 다시 매길 것이 없다.
        """
        if not (0 <= from_index < len(self._points)):
            return False
        if not (0 <= to_index < len(self._points)):
            return False
        if from_index == to_index:
            return False
        point = self._points.pop(from_index)
        self._points.insert(to_index, point)
        self._notify()
        return True

    def update_point(self, index, blob_setting=None, calibration=None):
        """콤보에서 고른 값을 담는다.

        표를 다시 그리지는 않는다. 값을 바꾼 콤보가 아직 신호를 처리하는
        중인데 그 콤보를 지우고 새로 만들면 안 되기 때문이다.
        """
        point = self.get_point(index)
        if point is None:
            return False
        if blob_setting is not None:
            point.blob_setting = blob_setting
        if calibration is not None:
            point.calibration = calibration
        # 다시 그리지는 않지만 '값이 바뀌었다'는 알려야 한다(모델 파일 저장).
        self._notify_data()
        return True

    def clear(self):
        if not self._points:
            return
        self._points.clear()
        self._notify()

    def set_points(self, points):
        self._points = list(points)
        self._notify()

    # ===== 변경 알림 =====
    def on_changed(self, callback):
        """표를 다시 그려야 하는 변경만 받는다."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def on_data_changed(self, callback):
        """값이 바뀐 모든 변경을 받는다(콤보 선택까지)."""
        if callback not in self._data_listeners:
            self._data_listeners.append(callback)

    def _notify(self):
        for callback in self._listeners:
            callback()
        self._notify_data()

    def _notify_data(self):
        for callback in self._data_listeners:
            callback()

    @staticmethod
    def from_list(data):
        model = FasteningDataModel()
        model._points = [FasteningPoint.from_dict(item) for item in data]
        return model
