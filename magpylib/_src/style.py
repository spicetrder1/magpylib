"""Collection of classes for display styling"""
# pylint: disable=C0302
import param

from magpylib._src.defaults.defaults_utility import (
    MagicParameterized,
    #color_validator,
    get_defaults_dict,
    validate_style_keys,
    SYMBOLS_MATPLOTLIB_TO_PLOTLY,
    LINESTYLES_MATPLOTLIB_TO_PLOTLY,
    MAGPYLIB_FAMILIES,
    SUPPORTED_PLOTTING_BACKENDS,
)


def get_style_class(obj):
    """returns style class based on object type. If class has no attribute `_object_type` or is
    not found in `MAGPYLIB_FAMILIES` returns `BaseStyle` class."""
    obj_type = getattr(obj, "_object_type", None)
    style_fam = MAGPYLIB_FAMILIES.get(obj_type, None)
    if isinstance(style_fam, (list, tuple)):
        style_fam = style_fam[0]
    return STYLE_CLASSES.get(style_fam, BaseStyle)


def get_style(obj, default_settings, **kwargs):
    """
    returns default style based on increasing priority:
    - style from defaults
    - style from object
    - style from kwargs arguments
    """
    # parse kwargs into style an non-style arguments
    style_kwargs = kwargs.get("style", {})
    style_kwargs.update(
        {k[6:]: v for k, v in kwargs.items() if k.startswith("style") and k != "style"}
    )

    # retrieve default style dictionary, local import to avoid circular import
    # pylint: disable=import-outside-toplevel

    styles_by_family = default_settings.display.style.as_dict()

    # construct object specific dictionary base on style family and default style
    obj_type = getattr(obj, "_object_type", None)
    obj_families = MAGPYLIB_FAMILIES.get(obj_type, [])

    obj_style_default_dict = {
        **styles_by_family["base"],
        **{
            k: v
            for fam in obj_families
            for k, v in styles_by_family.get(fam, {}).items()
        },
    }
    style_kwargs = validate_style_keys(style_kwargs)
    # create style class instance and update based on precedence
    obj_style = getattr(obj, "style", None)
    style = obj_style.copy() if obj_style is not None else BaseStyle()
    style_kwargs_specific = {
        k: v for k, v in style_kwargs.items() if k.split("_")[0] in style.as_dict()
    }
    style.update(**style_kwargs_specific, _match_properties=True)
    style.update(
        **obj_style_default_dict, _match_properties=False, _replace_None_only=True
    )

    return style


class Description(MagicParameterized):
    """Description styling properties"""

    show = param.Boolean(
        default=True,
        doc="if `True`, adds legend entry suffix based on value",
    )
    text = param.String(default=None, allow_None=True, doc="Object description text")

class Marker(MagicParameterized):
    """Defines the styling properties of plot markers"""

    color = param.Color(
            default=None,
            allow_None=True,
            doc="""The marker color. Must be a valid css color or one of
 `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
        )

    size = param.Number(
        default=None,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        softbounds=(1, 5),
        allow_None=True,
        doc="""Marker size""",
    )

    symbol = param.Selector(
        objects=[None] + list(SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys()),
        default=None,
        allow_None=True,
        doc = f"""Marker symbol. Can be one of: {list(SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys())}"""
    )


class Line(MagicParameterized):
    """Defines Line styling properties"""

    color = param.Color(default=None, allow_None=True, doc="""A valid css color""")

    width = param.Number(
        default=None,
        bounds=(0, 20),
        inclusive_bounds=(False, True),
        softbounds=(1, 5),
        allow_None=True,
        doc="""Path line width""",
    )

    style = param.Selector(
        objects=[None] + list(LINESTYLES_MATPLOTLIB_TO_PLOTLY.keys()),
        default=None,
        allow_None=True,
        doc = f"""Path line style. Can be one of: {list(LINESTYLES_MATPLOTLIB_TO_PLOTLY.keys())}"""
    )


class Path(MagicParameterized):
    """Defines the styling properties of an object's path"""

    marker = param.ClassSelector(
        Marker,
        default=Marker(),
        doc="""Marker class with `'color'``, 'symbol'`, `'size'` properties, or dictionary
with equivalent key/value pairs""",
    )

    line = param.ClassSelector(
        Line,
        default=Line(),
        doc="""Line class with `'color'``, 'width'`, `'style'` properties, or dictionary
with equivalent key/value pairs""",
    )

    show = param.Boolean(
        default=True,
        doc="""Show/hide path
- False: shows object(s) at final path position and hides paths lines and markers.
- True: shows object(s) shows object paths depending on `line`, `marker` and `frames`
parameters.""",
    )

    frames = param.List(
        item_type=int,
        doc= """Show copies of the 3D-model along the given path indices.
- integer i: displays the object(s) at every i'th path position.
- array_like shape (n,) of integers: describes certain path indices."""
    )

    numbering = param.Boolean(
        doc="""Show/hide numbering on path positions. Only applies if show=True.""",
    )

class Trace3d(MagicParameterized):
    """
    Defines properties for an additional user-defined 3d model object which is positioned relatively
    to the main object to be displayed and moved automatically with it. This feature also allows
    the user to replace the original 3d representation of the object.
    """

    show = param.Boolean(
        default=True,
        doc="""Shows/hides model3d object based on provided trace.""",
    )

    trace = param.Dict(
        instantiate=False,
        doc="""A dictionary or callable containing the parameters to build a trace for the chosen
backend."""
    )

    scale = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(False, False),
        softbounds=(0.1, 5),
        doc="""Scaling factor by which the trace vertices coordinates should be multiplied by.
Be aware that if the object is not centered at the global CS origin, its position will
also be scaled."""
    )

    backend = param.Selector(
        default='matplotlib',
        objects=list(SUPPORTED_PLOTTING_BACKENDS),
        doc=f"""Plotting backend corresponding to the trace. Can be one of
        {SUPPORTED_PLOTTING_BACKENDS}"""
    )

    coordsargs = param.Dict(
        instantiate=True,
        default={"x": "x", "y": "y", "z": "z"},
        doc="""Tells Magpylib the name of the coordinate arrays to be moved or rotated.
by default: `{"x": "x", "y": "y", "z": "z"}`"""
    )


class Model3d(MagicParameterized):
    """Defines properties for the 3d model representation of the magpylib object."""

    showdefault = param.Boolean(
        default=True,
        doc="""Shows/hides default 3D-model.""",
    )

    data = param.List(
        instantiate=True,
        item_type=Trace3d,
        doc="""A list of additional user-defined 3d model objects which is positioned relatively
to the main object to be displayed and moved automatically with it. This feature also allows
the user to replace the original 3d representation of the object"""
    )

    def add_trace(
        self, trace, show=True, backend="matplotlib", coordsargs=None, scale=1,
    ):
        """creates an additional user-defined 3d model object which is positioned relatively
        to the main object to be displayed and moved automatically with it. This feature also allows
        the user to replace the original 3d representation of the object

        Properties
        ----------
        show : bool, default=None
            Shows/hides model3d object based on provided trace:

        trace: dict or callable, default=None
            A dictionary or callable containing the parameters to build a trace for the chosen
            backend.

        backend:
            Plotting backend corresponding to the trace.
            Can be one of `['matplotlib', 'plotly']`

        coordsargs: dict
            Tells magpylib the name of the coordinate arrays to be moved or rotated.
            by default: `{"x": "x", "y": "y", "z": "z"}`
            if False, object is not rotated

        scale: float, default=1
            Scaling factor by which the trace vertices coordinates should be multiplied by. Be aware
            that if the object is not centered at the global CS origin, its position will
            also be scaled.
        """

        new_trace = Trace3d(
            trace=trace, show=show, scale=scale, backend=backend, coordsargs=coordsargs,
        )
        self.data = list(self.data) + [new_trace]
        return self

class BaseStyle(MagicParameterized):
    """Base class for display styling options of `BaseGeo` objects"""

    label = param.String(
        default=None,
        allow_None=True,
        doc="Label of the class instance, can be any string.",
    )
    description = param.ClassSelector(
        Description,
        default=Description(),
        doc="Object description properties such as `text` and `show`.",
    )
    color = param.Color(
        default=None,
        allow_None=True,
        doc="A valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.",
    )
    opacity = param.Number(
        default=None,
        allow_None=True,
        doc="Object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.",
    )
    path = param.ClassSelector(
        Path,
        default=Path(),
        doc="""An instance of `Path` or dictionary of equivalent key/value pairs, defining the
object path marker and path line properties.""",
    )
    model3d = param.ClassSelector(
        Model3d,
        default=Model3d(),
        doc=(
            """A list of traces where each is an instance of `Trace3d` or dictionary of equivalent
key/value pairs. Defines properties for an additional user-defined model3d object which is
positioned relatively to the main object to be displayed and moved automatically with it.
This feature also allows the user to replace the original 3d representation of the object."""
        ),
    )


class MagnetizationColor(MagicParameterized):
    """Defines the magnetization direction color styling properties."""

    _allowed_modes = ("bicolor", "tricolor", "tricycle")

    north = param.Color(
            default=None,
            allow_None=True,
            doc="""The color of the magnetic north pole. Must be a valid css color or one of
 `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
        )

    south = param.Color(
            default=None,
            allow_None=True,
            doc="""The color of the magnetic south pole. Must be a valid css color or one of
 `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
        )

    middle = param.Color(
            default=None,
            allow_None=True,
            doc="""The color between the magnetic poles. Must be a valid css color or one of
 `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
        )

    transition = param.Number(
        default=None,
        bounds=(0, 1),
        inclusive_bounds=(True, True),
        allow_None=True,
        doc="""Sets the transition smoothness between poles colors. Must be between 0 and 1.
- `transition=0`: discrete transition
- `transition=1`: smoothest transition
        """
    )

    mode = param.Selector(
        default='tricolor',
        objects=_allowed_modes,
        doc="""Sets the coloring mode for the magnetization.
- `'bicolor'`: only north and south poles are shown, middle color is hidden.
- `'tricolor'`: both pole colors and middle color are shown.
- `'tricycle'`: both pole colors are shown and middle color is replaced by a color cycling
through the color sequence."""
    )


class Magnetization(MagicParameterized):
    """Defines magnetization styling properties"""

    show = param.Boolean(
        default=True,
        doc="""Show/hide magnetization based on active plotting backend""",
    )

    size = param.Number(
        default=None,
        bounds=(0, None),
        inclusive_bounds=(True, True),
        softbounds=(1, 5),
        allow_None=True,
        doc="""Arrow size of the magnetization direction (for the matplotlib backend only),
only applies if `show=True`""",
    )

    color = param.ClassSelector(
        MagnetizationColor,
        default=MagnetizationColor(),
        doc="""Color properties showing the magnetization direction (for the plotly backend),
only applies if `show=True`"""
    )

class MagnetSpecific(MagicParameterized):
    """Defines the specific styling properties of objects of the `Magnet` family."""

    magnetization = param.ClassSelector(
        Magnetization,
        default=Magnetization(),
        doc="""Magnetization styling with `'show'`, `'size'`, `'color'` properties or a dictionary
with equivalent key/value pairs"""
    )
class MagnetStyle(BaseStyle, MagnetSpecific):
    """Defines the styling properties of objects of the `Magnet` family."""


class ArrowSingle(MagicParameterized):
    """Single coordinate system arrow properties"""

    show = param.Boolean(
        default=True,
        doc="""Show/hide single CS arrow""",
    )

    param.Color(
            default=None,
            allow_None=True,
            doc="""The color of a single CS arrow. Must be a valid css color or one of
 `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
        )

class ArrowCS(MagicParameterized):
    """Triple coordinate system arrow properties"""

    x = param.ClassSelector(
        ArrowSingle,
        default=ArrowSingle(),
        doc="""`Arrowsingle` class or dict with equivalent key/value pairs for x-direction
        (e.g. `color`, `show`)"""
    )

    y = param.ClassSelector(
        ArrowSingle,
        default=ArrowSingle(),
        doc="""`Arrowsingle` class or dict with equivalent key/value pairs for y-direction
        (e.g. `color`, `show`)"""
    )

    z = param.ClassSelector(
        ArrowSingle,
        default=ArrowSingle(),
        doc="""`Arrowsingle` class or dict with equivalent key/value pairs for z-direction
        (e.g. `color`, `show`)"""
    )

class Pixel(MagicParameterized):
    """Defines the styling properties of sensor pixels"""

    size = param.Number(
        default=None,
        bounds=(0, None),
        inclusive_bounds=(True, None),
        softbounds=(0.5, 2),
        allow_None=True,
        doc="""The relative pixel size.
- matplotlib backend: pixel size is the marker size
- plotly backend:  relative size to the distance of nearest neighboring pixel""",
    )

    param.Color(
            default=None,
            allow_None=True,
            doc="""The color of sensor pixel. Must be a valid css color or one of
 `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
        )

    symbol = param.Selector(
        objects=[None] + list(SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys()),
        default=None,
        allow_None=True,
        doc = f"""Marker symbol. Can be one of: {list(SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys())}"""
    )
class SensorSpecific(MagicParameterized):
    """ Defines the specific styling properties of objects of the `sensor` family"""

    size = param.Number(
        default=None,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        softbounds=(1, 5),
        allow_None=True,
        doc="""Sensor size relative to the canvas size.""",
    )

    arrows = param.ClassSelector(
        ArrowCS,
        default=ArrowCS(),
        doc="""`ArrowCS` class or dict with equivalent key/value pairs (e.g. `color`, `size`)"""
    )

    pixel = param.ClassSelector(
        Pixel,
        default=Pixel(),
        doc="""`Pixel` class or dict with equivalent key/value pairs (e.g. `color`, `size`)"""
    )

class SensorStyle(BaseStyle, SensorSpecific):
    """Defines the styling properties of objects of the `sensor` family"""



class CurentArrow(MagicParameterized):
    """Defines the styling properties of current arrows."""

    show = param.Boolean(
        default=True,
        doc="""Show/hide current direction arrow""",
    )

    size = param.Number(
        default=None,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        softbounds=(0.5, 5),
        allow_None=True,
        doc="""The current arrow size""",
    )

    width = param.Number(
        default=None,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        softbounds=(0.5, 5),
        allow_None=True,
        doc="""The current arrow width""",
    )

class CurrentSpecific(MagicParameterized):
    """Defines the specific styling properties of objects of the `current` family."""

    arrow = param.ClassSelector(
        CurentArrow,
        default=CurentArrow(),
        doc="""`CurentArrow` class or dict with equivalent key/value pairs
(e.g. `'show'`, `'size')"""
    )

class CurrentStyle(BaseStyle, CurrentSpecific):
    """Defines the styling properties of objects of the `current` family."""


class MarkersStyle(BaseStyle):
    """Defines the styling properties of the markers trace."""
    marker = param.ClassSelector(
            Marker,
            default=Marker(),
            doc="""Marker class with `'color'``, 'symbol'`, `'size'` properties, or dictionary
    with equivalent key/value pairs""",
        )


class DipoleSpecific(MagicParameterized):
    """Defines the specific styling properties of the objects of the `dipole` family"""

    _allowed_pivots = ("tail", "middle", "tip")

    size = param.Number(
        default=None,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        softbounds=(0.5, 5),
        allow_None=True,
        doc="""The dipole arrow size relative to the canvas size""",
    )

    pivot = param.Selector(
        default='middle',
        objects=_allowed_pivots,
        doc="""The part of the arrow that is anchored to the X, Y grid.
The arrow rotates about this point. Can be one of `['tail', 'middle', 'tip']`"""
    )
class DipoleStyle(BaseStyle, DipoleSpecific):
    """Defines the styling properties of the objects of the `dipole` family."""

class DisplayStyle(MagicParameterized):
    """
    Base class containing styling properties for all object families. The properties of the
    sub-classes get set to hard coded defaults at class instantiation.
    """

    def reset(self):
        """Resets all nested properties to their hard coded default values"""
        self.update(get_defaults_dict("display.style"), _match_properties=False)
        return self

    base = param.ClassSelector(
        BaseStyle,
        default=BaseStyle(),
        doc="""Base properties common to all families"""
    )

    magnet=param.ClassSelector(
        MagnetSpecific,
        default=MagnetSpecific(),
        doc="""Magnet properties"""
    )

    current=param.ClassSelector(
        CurrentSpecific,
        default=CurrentSpecific(),
        doc="""Current properties"""
    )

    dipole=param.ClassSelector(
        DipoleSpecific,
        default=DipoleSpecific(),
        doc="""Dipole properties"""
    )

    sensor=param.ClassSelector(
        SensorSpecific,
        default=SensorSpecific(),
        doc="""Sensor properties"""
    )

    markers=param.ClassSelector(
        MarkersStyle,
        default=MarkersStyle(),
        doc="""Markers properties"""
    )


STYLE_CLASSES = {
    "magnet": MagnetStyle,
    "current": CurrentStyle,
    "dipole": DipoleStyle,
    "sensor": SensorStyle,
    "markers": MarkersStyle
}
