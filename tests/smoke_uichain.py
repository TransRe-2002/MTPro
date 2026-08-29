"""Probe: which ui.* module import crashes (stage-printed)."""
import sys

print("C0 start", flush=True)
import ui.log_console

print("C1 ui.log_console ok", flush=True)
import ui.data_tree_view

print("C2 ui.data_tree_view ok", flush=True)
import ui.data_process

print("C3 ui.data_process ok", flush=True)
import ui.data_view_widget

print("C4 ui.data_view_widget ok", flush=True)
print("UICH_SMOKE_OK", flush=True)
sys.exit(0)