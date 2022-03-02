"""BaseGeo class code
READY FOR V4
"""

from magpylib._src.display.display import show

class BaseDisplayRepr:
    """Provides the display(self) and self.repr methods for all objects"""

    show = show
    _object_type = None

    def __repr__(self) -> str:
        name = getattr(self, "name", None)
        if name is None and hasattr(self, "style"):
            name = getattr(getattr(self, "style"), "label", "").strip()
        name_str =  f", label={name!r}" if name!="" else name
        return f"{type(self).__name__}(id={id(self)!r}{name_str!r})"
