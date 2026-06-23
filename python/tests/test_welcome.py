from PySide6.QtCore import Qt

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


def test_welcome_recent_file_activation_emits_path(qtbot, tmp_path):
    recent = tmp_path / "recent.mat"
    widget = Welcome()
    qtbot.addWidget(widget)
    widget.set_recent_files([str(recent)])

    emitted: list[str] = []
    widget.recent_file_requested.connect(emitted.append)
    widget._open_recent_index(widget.recent_model.index(0, 0))

    assert emitted == [str(recent)]


def test_welcome_recent_files_placeholder_when_empty(qtbot):
    widget = Welcome()
    qtbot.addWidget(widget)
    widget.set_recent_files([])
    widget.show()
    qtbot.waitExposed(widget)

    assert widget.recent_placeholder.isVisible() is True
    assert widget.recent_view.isVisible() is False
    assert widget.recent_model.rowCount() == 0
