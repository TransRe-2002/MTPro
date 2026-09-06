"""First-party and third-party MTPro processing plugin packages."""

# Imported lazily to avoid a package cycle while individual feature packages
# are importing their editors.
def __getattr__(name):
    if name == "BUILTIN_PLUGINS":
        from builtin_plugins import BUILTIN_PLUGINS
        return BUILTIN_PLUGINS
    raise AttributeError(name)

__all__ = ["BUILTIN_PLUGINS"]
