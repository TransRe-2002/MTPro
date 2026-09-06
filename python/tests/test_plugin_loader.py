from __future__ import annotations

from pathlib import Path
import sys
import zipfile

from base.plugin_loader import load_all_plugins


PLUGIN_MODULE = '''
from plugin_api import ParamSpec, PluginInfo, TransformResult

class ExamplePlugin:
    info = PluginInfo(
        plugin_id={plugin_id!r},
        display_name={name!r},
        version="1.0.0",
        params_schema=(ParamSpec("offset", "Offset", float, 0.0),),
    )

    def process(self, block, params, ctx):
        return TransformResult(block.series + params["offset"])
'''


def _manifest(plugin_id: str, name: str, module: str, api: int = 1) -> str:
    return f'''[plugin]
id = "{plugin_id}"
name = "{name}"
version = "1.0.0"
api = {api}
entry = "{module}:ExamplePlugin"
'''


def _write_plugin(root: Path, folder: str, plugin_id: str, name: str) -> None:
    candidate = root / folder
    package = candidate / f"{folder}_pkg"
    package.mkdir(parents=True)
    (candidate / "plugin.toml").write_text(
        _manifest(plugin_id, name, package.name),
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        PLUGIN_MODULE.format(plugin_id=plugin_id, name=name),
        encoding="utf-8",
    )


def test_directory_loader_isolates_failures_and_rejects_duplicate_ids(tmp_path):
    _write_plugin(tmp_path, "a_good", "example.good", "Good")
    _write_plugin(tmp_path, "b_duplicate", "example.good", "Good")

    bad_entry = tmp_path / "c_bad_entry"
    bad_entry.mkdir()
    (bad_entry / "plugin.toml").write_text(
        _manifest("example.bad-entry", "Bad entry", "missing_package"),
        encoding="utf-8",
    )

    bad_api = tmp_path / "d_bad_api"
    bad_api.mkdir()
    (bad_api / "plugin.toml").write_text(
        _manifest("example.bad-api", "Bad API", "anything", api=99),
        encoding="utf-8",
    )

    report = load_all_plugins(
        scan_paths=[tmp_path],
        builtin_classes=[],
        include_entry_points=False,
    )

    assert [item.info.plugin_id for item in report.loaded] == ["example.good"]
    failed_ids = {failure.plugin_id for failure in report.failed}
    assert failed_ids == {"example.good", "example.bad-entry", "example.bad-api"}
    assert any("duplicate plugin_id" in failure.reason for failure in report.failed)


def test_zip_plugin_loads_from_root_manifest(tmp_path):
    archive_path = tmp_path / "zip_plugin.zip"
    module_name = "zip_example_pkg"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("plugin.toml", _manifest("example.zip", "Zip", module_name))
        archive.writestr(
            f"{module_name}/__init__.py",
            PLUGIN_MODULE.format(plugin_id="example.zip", name="Zip"),
        )

    report = load_all_plugins(
        scan_paths=[tmp_path],
        builtin_classes=[],
        include_entry_points=False,
    )

    assert [item.info.plugin_id for item in report.loaded] == ["example.zip"]
    assert report.loaded[0].source == f"zip:{archive_path}"
    assert report.failed == []
    assert str(archive_path) not in sys.path


def test_builtin_precedence_wins_over_scanned_plugin(tmp_path):
    from plugin_api import PluginInfo, TransformResult

    class Builtin:
        info = PluginInfo("example.same", "Builtin", "1.0.0")

        def process(self, block, params, ctx):
            return TransformResult(block.series)

    _write_plugin(tmp_path, "external", "example.same", "External")

    report = load_all_plugins(
        scan_paths=[tmp_path],
        builtin_classes=[Builtin],
        include_entry_points=False,
    )

    assert [(item.info.plugin_id, item.source) for item in report.loaded] == [
        ("example.same", "builtin:Builtin")
    ]
    assert len(report.failed) == 1
    assert "duplicate plugin_id" in report.failed[0].reason


def test_manifest_id_must_match_plugin_info(tmp_path):
    _write_plugin(tmp_path, "mismatch", "example.manifest", "Mismatch")
    module_path = tmp_path / "mismatch" / "mismatch_pkg" / "__init__.py"
    module_path.write_text(
        PLUGIN_MODULE.format(plugin_id="example.code", name="Mismatch"),
        encoding="utf-8",
    )

    report = load_all_plugins(
        scan_paths=[tmp_path],
        builtin_classes=[],
        include_entry_points=False,
    )

    assert report.loaded == []
    assert report.failed[0].plugin_id == "example.manifest"
    assert "does not match" in report.failed[0].reason
