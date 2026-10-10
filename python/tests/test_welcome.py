from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QAbstractItemView

from ui.welcome import Welcome


def test_welcome_recent_files_use_tree_view(qtbot, tmp_path):
    first = tmp_path / "first.mat"
    second = tmp_path / "second.mat"

    widget = Welcome()
    qtbot.addWidget(widget)
    widget.set_recent_files([str(first), str(second)])

    assert widget.recent_view.isVisible() is False
    widget.show()
    qtbot.waitExposed(widget)

    assert widget.recent_placeholder.isVisible() is False
    assert widget.recent_view.isVisible() is True
    assert widget.recent_model.rowCount() == 2
    assert widget.recent_model.item(0, 0).text() == "first.mat"
    assert widget.recent_model.item(0, 1).text() == str(tmp_path)
    assert widget.recent_model.item(0, 0).data(Qt.ItemDataRole.UserRole) == str(first)
    assert widget.recent_view.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert widget.open_recent_button.isVisible() is True
    assert widget.open_recent_button.isEnabled() is False


def test_welcome_recent_file_activation_emits_path(qtbot, tmp_path):
    recent = tmp_path / "recent.mat"
    widget = Welcome()
    qtbot.addWidget(widget)
    widget.set_recent_files([str(recent)])

    emitted: list[str] = []
    widget.recent_file_requested.connect(emitted.append)
    widget._open_recent_index(widget.recent_model.index(0, 0))

    assert emitted == [str(recent)]


def test_welcome_opens_multiple_selected_recent_files_in_row_order(qtbot, tmp_path):
    first = tmp_path / "first.mat"
    second = tmp_path / "second.mat"
    third = tmp_path / "third.mat"
    widget = Welcome()
    qtbot.addWidget(widget)
    widget.set_recent_files([str(first), str(second), str(third)])

    emitted: list[list[str]] = []
    widget.recent_files_requested.connect(emitted.append)
    selection_model = widget.recent_view.selectionModel()
    flags = (
        QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows
    )
    selection_model.select(widget.recent_model.index(2, 0), flags)
    selection_model.select(widget.recent_model.index(0, 0), flags)

    assert widget.open_recent_button.text() == "打开所选文件（2）"
    widget.open_recent_button.click()

    assert emitted == [[str(first), str(third)]]


def test_welcome_recent_files_placeholder_when_empty(qtbot):
    widget = Welcome()
    qtbot.addWidget(widget)
    widget.set_recent_files([])
    widget.show()
    qtbot.waitExposed(widget)

    assert widget.recent_placeholder.isVisible() is True
    assert widget.recent_view.isVisible() is False
    assert widget.recent_model.rowCount() == 0
