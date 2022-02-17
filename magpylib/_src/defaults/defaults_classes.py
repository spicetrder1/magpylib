import param
import numpy as np
from magpylib._src.defaults.defaults_utility import (
    MagicParameterized,
    color_validator,
    get_defaults_dict,
    SUPPORTED_PLOTTING_BACKENDS,
    SYMBOLS_MATPLOTLIB_TO_PLOTLY,
    LINESTYLES_MATPLOTLIB_TO_PLOTLY,
)


class Description(MagicParameterized):
    """Description styling properties"""

    show = param.Boolean(
        default=True, doc="if `True`, adds legend entry suffix based on value",
    )
    text = param.String(doc="Object description text")


class Marker(MagicParameterized):
    """Defines the styling properties of plot markers"""

    color = param.Color(
        default=None,
        allow_None=True,
        doc="""
        The marker color. Must be a valid css color or one of
        `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
    )

    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, True),
        softbounds=(1, 5),
        doc="""Marker size""",
    )

    symbol = param.Selector(
        objects=list(SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys()),
        doc=f"""Marker symbol. Can be one of: {list(SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys())}""",
    )

class PathMarker(Marker):
    """Defines the styling properties of path plot markers"""


class Line(MagicParameterized):
    """Defines Line styling properties"""

    color = param.Color(default=None, allow_None=True, doc="""A valid css color""")

    width = param.Number(
        default=1,
        bounds=(0, 20),
        inclusive_bounds=(True, True),
        softbounds=(1, 5),
        doc="""Path line width""",
    )

    style = param.Selector(
        default="solid",
        objects=list(LINESTYLES_MATPLOTLIB_TO_PLOTLY.keys()),
        doc=f"""
        Path line style. Can be one of:
        {list(LINESTYLES_MATPLOTLIB_TO_PLOTLY.keys())}""",
    )


class Frames(MagicParameterized):
    """Defines the styling properties of an object's path frames"""

    indices = param.List(
        default=[],
        item_type=int,
        doc="""Array_like shape (n,) of integers: describes certain path indices.""",
    )

    step = param.Integer(
        default=0, doc="""Displays the object(s) at every i'th path position"""
    )

    mode = param.Selector(
        default="indices",
        objects=["indices", "step"],
        doc="""
        The object path frames mode.
        - step: integer i: displays the object(s) at every i'th path position.
        - indices: array_like shape (n,) of integers: describes certain path indices.""",
    )

    @param.depends("indices", watch=True)
    def _update_indices(self):
        self.mode = "indices"

    @param.depends("step", watch=True)
    def _update_step(self):
        self.mode = "step"


class Path(MagicParameterized):
    """Defines the styling properties of an object's path"""

    def __setattr__(self, name, value):
        if name == "frames":
            if isinstance(value, (tuple, list, np.ndarray)):
                self.frames.indices = [int(v) for v in value]
            elif isinstance(value, (int, np.integer)):
                self.frames.step = value
            else:
                super().__setattr__(name, value)
            return
        super().__setattr__(name, value)

    marker = param.ClassSelector(
        Marker,
        default=PathMarker(),
        doc="""
        Marker class with `'color'``, 'symbol'`, `'size'` properties, or dictionary with equivalent
        key/value pairs""",
    )

    line = param.ClassSelector(
        Line,
        default=Line(),
        doc="""
        Line class with `'color'``, 'width'`, `'style'` properties, or dictionary with equivalent
        key/value pairs""",
    )

    show = param.Boolean(
        default=True,
        doc="""
        Show/hide path
        - False: shows object(s) at final path position and hides paths lines and markers.
        - True: shows object(s) shows object paths depending on `line`, `marker` and `frames`
                parameters.""",
    )

    numbering = param.Boolean(
        doc="""Show/hide numbering on path positions. Only applies if show=True.""",
    )

    frames = param.ClassSelector(
        Frames,
        default=Frames(),
        doc="""
        Show copies of the 3D-model along the given path indices.
        - mode: either `step` or `indices`.
        - step: integer i: displays the object(s) at every i'th path position.
        - indices: array_like shape (n,) of integers: describes certain path indices.""",
    )


class Trace3d(MagicParameterized):
    """
    Defines properties for an additional user-defined 3d model object which is positioned relatively
    to the main object to be displayed and moved automatically with it. This feature also allows
    the user to replace the original 3d representation of the object.
    """

    def __setattr__(self, name, value):
        if name == "coordsargs":
            if value is None:
                value = {"x": "x", "y": "y", "z": "z"}
            assert isinstance(value, dict) and all(key in value for key in "xyz"), (
                f"the `coordsargs` property of {type(self).__name__} must be "
                f"a dictionary with `'x', 'y', 'z'` keys"
                f" but received {repr(value)} instead"
            )
        elif name == "trace":
            assert isinstance(value, dict) or callable(value), (
                "trace must be a dictionary or a callable returning a dictionary"
                f" but received {type(value).__name__} instead"
            )
            test_val = value
            if callable(value):
                test_val = value()
            assert "type" in test_val, "explicit trace `type` must be defined"
        return super().__setattr__(name, value)

    show = param.Boolean(
        default=True, doc="""Shows/hides model3d object based on provided trace.""",
    )

    trace = param.Parameter(
        doc="""
        A dictionary or callable containing the parameters to build a trace for the chosen
        backend.""",
    )

    scale = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, False),
        softbounds=(0.1, 5),
        doc="""
        Scaling factor by which the trace vertices coordinates should be multiplied by.
        Be aware that if the object is not centered at the global CS origin, its position will also
        be scaled.""",
    )

    backend = param.Selector(
        default="matplotlib",
        objects=list(SUPPORTED_PLOTTING_BACKENDS),
        doc=f"""
        Plotting backend corresponding to the trace. Can be one of
        {SUPPORTED_PLOTTING_BACKENDS}""",
    )

    coordsargs = param.Dict(
        default={"x": "x", "y": "y", "z": "z"},
        doc="""
        Tells Magpylib the name of the coordinate arrays to be moved or rotated.
        by default: `{"x": "x", "y": "y", "z": "z"}`""",
    )


class Model3d(MagicParameterized):
    """Defines properties for the 3d model representation of the magpylib object."""

    def __setattr__(self, name, value):
        if name == "data":
            value = [Trace3d(**v) if isinstance(v, dict) else v for v in value]
        return super().__setattr__(name, value)

    showdefault = param.Boolean(default=True, doc="""Shows/hides default 3D-model.""",)

    data = param.List(
        item_type=Trace3d,
        doc="""
        A list of additional user-defined 3d model objects which is positioned relatively to the
        main object to be displayed and moved automatically with it. This feature also allows the
        user to replace the original 3d representation of the object""",
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

    label = param.String(doc="Label of the class instance, can be any string.")
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
        default=1,
        bounds=(0, 1),
        doc="Object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.",
    )
    path = param.ClassSelector(
        Path,
        default=Path(),
        doc="""
        An instance of `Path` or dictionary of equivalent key/value pairs, defining the object path
        marker and path line properties.""",
    )
    model3d = param.ClassSelector(
        Model3d,
        default=Model3d(),
        doc=(
            """
        A list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.
        """
        ),
    )


class MagnetizationColor(MagicParameterized):
    """Defines the magnetization direction color styling properties."""

    _allowed_modes = ("bicolor", "tricolor", "tricycle")

    north = param.Color(
        default="red",
        doc="""
        The color of the magnetic north pole. Must be a valid css color or one of
        `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
    )

    south = param.Color(
        default="green",
        doc="""
        The color of the magnetic south pole. Must be a valid css color or one of
        `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
    )

    middle = param.Color(
        default="grey",
        doc="""
        The color between the magnetic poles. Must be a valid css color or one of
        `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
    )

    transition = param.Number(
        default=0.2,
        bounds=(0, 1),
        inclusive_bounds=(True, True),
        doc="""
        Sets the transition smoothness between poles colors. Must be between 0 and 1.
        - `transition=0`: discrete transition
        - `transition=1`: smoothest transition
        """,
    )

    mode = param.Selector(
        default="tricolor",
        objects=_allowed_modes,
        doc="""
        Sets the coloring mode for the magnetization.
        - `'bicolor'`: only north and south poles are shown, middle color is hidden.
        - `'tricolor'`: both pole colors and middle color are shown.
        - `'tricycle'`: both pole colors are shown and middle color is replaced by a color cycling
        through the color sequence.""",
    )


class Magnetization(MagicParameterized):
    """Defines magnetization styling properties"""

    show = param.Boolean(
        default=True,
        doc="""Show/hide magnetization based on active plotting backend""",
    )

    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, True),
        softbounds=(1, 5),
        doc="""
        Arrow size of the magnetization direction (for the matplotlib backend only), only applies if
        `show=True`""",
    )

    color = param.ClassSelector(
        MagnetizationColor,
        default=MagnetizationColor(),
        doc="""
        Color properties showing the magnetization direction (for the plotly backend), only applies
        if `show=True`""",
    )


class MagnetSpecific(MagicParameterized):
    """Defines the specific styling properties of objects of the `Magnet` family."""

    magnetization = param.ClassSelector(
        Magnetization,
        default=Magnetization(),
        doc="""
        Magnetization styling with `'show'`, `'size'`, `'color'` properties or a dictionary with
        equivalent key/value pairs""",
    )


class MagnetStyle(BaseStyle, MagnetSpecific):
    """Defines the styling properties of objects of the `Magnet` family."""


class ArrowSingle(MagicParameterized):
    """Single coordinate system arrow properties"""

    show = param.Boolean(default=True, doc="""Show/hide single CS arrow""",)

    color = param.Color(
        default=None,
        allow_None=True,
        doc="""
        The color of a single CS arrow. Must be a valid css color or one of
        `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
    )


class ArrowX(ArrowSingle):
    """Single coordinate system x-arrow properties"""


class ArrowY(ArrowSingle):
    """Single coordinate system y-arrow properties"""


class ArrowZ(ArrowSingle):
    """Single coordinate system z-arrow properties"""


class ArrowCS(MagicParameterized):
    """Triple coordinate system arrow properties"""

    x = param.ClassSelector(
        ArrowX,
        default=ArrowX(),
        doc="""
        `Arrowsingle` class or dict with equivalent key/value pairs for x-direction
        (e.g. `color`, `show`)""",
    )

    y = param.ClassSelector(
        ArrowY,
        default=ArrowY(),
        doc="""
        `Arrowsingle` class or dict with equivalent key/value pairs for y-direction
        (e.g. `color`, `show`)""",
    )

    z = param.ClassSelector(
        ArrowZ,
        default=ArrowZ(),
        doc="""
        `Arrowsingle` class or dict with equivalent key/value pairs for z-direction
        (e.g. `color`, `show`)""",
    )


class Pixel(MagicParameterized):
    """Defines the styling properties of sensor pixels"""

    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, None),
        softbounds=(0.5, 2),
        doc="""
        The relative pixel size.
        - matplotlib backend: pixel size is the marker size
        - plotly backend:  relative size to the distance of nearest neighboring pixel""",
    )

    color = param.Color(
        default=None,
        allow_None=True,
        doc="""
        The color of sensor pixel. Must be a valid css color or one of
        `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
    )

    symbol = param.Selector(
        default="o",
        objects=list(SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys()),
        doc=f"""
        Marker symbol. Can be one of:
        {list(SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys())}""",
    )


class SensorSpecific(MagicParameterized):
    """Defines the specific styling properties of objects of the `sensor` family"""

    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, True),
        softbounds=(1, 5),
        doc="""Sensor size relative to the canvas size.""",
    )

    arrows = param.ClassSelector(
        ArrowCS,
        default=ArrowCS(),
        doc="""`ArrowCS` class or dict with equivalent key/value pairs (e.g. `color`, `size`)""",
    )

    pixel = param.ClassSelector(
        Pixel,
        default=Pixel(),
        doc="""`Pixel` class or dict with equivalent key/value pairs (e.g. `color`, `size`)""",
    )


class SensorStyle(BaseStyle, SensorSpecific):
    """Defines the styling properties of objects of the `sensor` family"""


class CurentArrow(MagicParameterized):
    """Defines the styling properties of current arrows."""

    show = param.Boolean(default=True, doc="""Show/hide current direction arrow""",)

    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, True),
        softbounds=(0.5, 5),
        doc="""The current arrow size""",
    )

    width = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, None),
        softbounds=(0.5, 5),
        doc="""The current arrow width""",
    )


class CurrentSpecific(MagicParameterized):
    """Defines the specific styling properties of objects of the `current` family."""

    arrow = param.ClassSelector(
        CurentArrow,
        default=CurentArrow(),
        doc="""
        `CurentArrow` class or dict with equivalent key/value pairs
        (e.g. `'show'`, `'size')""",
    )


class CurrentStyle(BaseStyle, CurrentSpecific):
    """Defines the styling properties of objects of the `current` family."""


class MarkersStyle(BaseStyle):
    """Defines the styling properties of the markers trace."""

    marker = param.ClassSelector(
        Marker,
        default=Marker(),
        doc="""
        Marker class with `'color'``, 'symbol'`, `'size'` properties, or dictionary with equivalent
        key/value pairs""",
    )


class DipoleSpecific(MagicParameterized):
    """Defines the specific styling properties of the objects of the `dipole` family"""

    _allowed_pivots = ("tail", "middle", "tip")

    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, True),
        softbounds=(0.5, 5),
        doc="""The dipole arrow size relative to the canvas size""",
    )

    pivot = param.Selector(
        default="middle",
        objects=_allowed_pivots,
        doc="""The part of the arrow that is anchored to the X, Y grid. The arrow rotates about
        this point. Can be one of `['tail', 'middle', 'tip']`""",
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
        BaseStyle, default=BaseStyle(), doc="""Base properties common to all families"""
    )

    magnet = param.ClassSelector(
        MagnetSpecific, default=MagnetSpecific(), doc="""Magnet properties"""
    )

    current = param.ClassSelector(
        CurrentSpecific, default=CurrentSpecific(), doc="""Current properties"""
    )

    dipole = param.ClassSelector(
        DipoleSpecific, default=DipoleSpecific(), doc="""Dipole properties"""
    )

    sensor = param.ClassSelector(
        SensorSpecific, default=SensorSpecific(), doc="""Sensor properties"""
    )

    markers = param.ClassSelector(
        MarkersStyle, default=MarkersStyle(), doc="""Markers properties"""
    )


class Animation(MagicParameterized):
    """
    Defines the animation properties used by the `plotly` plotting backend when `animation=True`
    in the `display` function.
    """

    fps = param.Integer(
        default=30,
        bounds=(0, None),
        inclusive_bounds=(False, None),
        doc="""Target number of frames to be displayed per second.""",
    )

    maxfps = param.Integer(
        default=50,
        bounds=(0, None),
        inclusive_bounds=(False, None),
        doc="""Maximum number of frames to be displayed per second before downsampling kicks in.""",
    )

    maxframes = param.Integer(
        default=200,
        bounds=(0, None),
        inclusive_bounds=(False, None),
        doc="""Maximum total number of frames to be displayed before downsampling kicks in.""",
    )

    time = param.Number(
        default=5,
        bounds=(0, None),
        inclusive_bounds=(False, None),
        doc="""Default animation time.""",
    )

    slider = param.Boolean(
        default=True, doc="""Show/hide an interactive animation slider""",
    )


class Display(MagicParameterized):
    """Defines the display properties for the plotting features"""

    def __setattr__(self, name, value):
        if name == "colorsequence":
            value = [
                color_validator(v, allow_None=False, parent_name="Colorsequence")
                for v in value
            ]
        return super().__setattr__(name, value)

    backend = param.Selector(
        default="matplotlib",
        objects=list(SUPPORTED_PLOTTING_BACKENDS),
        doc=f"""
        Plotting backend corresponding to the trace. Can be one of
        {SUPPORTED_PLOTTING_BACKENDS}""",
    )

    colorsequence = param.List(
        default=[
            "#2E91E5",
            "#E15F99",
            "#1CA71C",
            "#FB0D0D",
            "#DA16FF",
            "#222A2A",
            "#B68100",
            "#750D86",
            "#EB663B",
            "#511CFB",
            "#00A08B",
            "#FB00D1",
            "#FC0080",
            "#B2828D",
            "#6C7C32",
            "#778AAE",
            "#862A16",
            "#A777F1",
            "#620042",
            "#1616A7",
            "#DA60CA",
            "#6C4516",
            "#0D2A63",
            "#AF0038",
        ],
        doc="""
        A list of color values used to cycle trough for every object displayed.
        A color and may be specified as:
        - An rgb string (e.g. 'rgb(255,0,0)')
        - A named CSS color (e.g. 'magenta')
        - A hex string (e.g. '#B68100')""",
    )

    animation = param.ClassSelector(
        Animation,
        default=Animation(),
        doc="""
        The animation properties used when `animation=True` in the `show` function. This settings
        only apply to the `plotly` plotting backend for the moment""",
    )

    autosizefactor = param.Number(
        default=10,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        softbounds=(5, 15),
        doc="""
        Defines at which scale objects like sensors and dipoles are displayed.
        -> object_size = canvas_size / AUTOSIZE_FACTOR""",
    )

    style = param.ClassSelector(
        DisplayStyle,
        default=DisplayStyle(),
        doc="""Base display default-class containing styling properties for all object families.""",
    )


class DefaultConfig(MagicParameterized):
    """Library default settings. All default values get reset at class instantiation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._declare_watchers()
        with param.parameterized.batch_call_watchers(self):
            self.reset()

    def reset(self):
        """Resets all nested properties to their hard coded default values"""
        self.update(get_defaults_dict(), _match_properties=False)
        return self

    def _declare_watchers(self):
        props = get_defaults_dict(flatten=True, separator=".").keys()
        for prop in props:
            attrib_chain = prop.split(".")
            child = attrib_chain[-1]
            parent = self  # start with self to go through dot chain
            for attrib in attrib_chain[:-1]:
                parent = getattr(parent, attrib)
            parent.param.watch(self._set_to_defaults, parameter_names=[child])

    @staticmethod
    def _set_to_defaults(event):
        """Sets class defaults whenever magpylib defaults attributes as set"""
        event.obj.param.set_default(event.name, event.new)

    checkinputs = param.Boolean(
        default=True,
        doc="""
        Check user input types, shapes at various stages and raise errors when they are not within
        the designated constrains.""",
    )

    display = param.ClassSelector(
        Display,
        default=Display(),
        doc="""
        `Display` defaults-class containing display settings.
        `(e.g. 'backend', 'animation', 'colorsequence', ...)`""",
    )


default_settings = DefaultConfig()
