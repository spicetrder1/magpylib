import param
from magpylib._src.defaults.defaults_utility import (
    MagicParameterized,
    color_validator,
    # color_validator,
    get_defaults_dict,
    SUPPORTED_PLOTTING_BACKENDS,
)
from magpylib._src.style import DisplayStyle


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
        default=True,
        doc="""Show/hide an interactive animation slider""",
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
        doc=f"""Plotting backend corresponding to the trace. Can be one of
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
        doc="""A list of color values used to cycle trough for every object displayed.
        A color and may be specified as:
      - An rgb string (e.g. 'rgb(255,0,0)')
      - A named CSS color (e.g. 'magenta')
      - A hex string (e.g. '#B68100')""",
    )

    animation = param.ClassSelector(
        Animation,
        default=Animation(),
        doc="""The animation properties used when `animation=True` in the `show` function.
This settings only apply to the `plotly` plotting backend for the moment""",
    )

    autosizefactor = param.Number(
        default=10,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        softbounds=(5, 15),
        doc="""Defines at which scale objects like sensors and dipoles are displayed.
-> object_size = canvas_size / AUTOSIZE_FACTOR""",
    )

    style = param.ClassSelector(
        DisplayStyle,
        default=DisplayStyle(),
        doc="""Base display default-class containing styling properties for all object families.""",
    )


class DefaultConfig(MagicParameterized):
    """Library default settings. All default values get reset at class instantiation."""

    def __init__(self, *args, reset=True, **kwargs):
        super().__init__(*args, **kwargs)
        if reset:
            self.reset()

    def reset(self):
        """Resets all nested properties to their hard coded default values"""
        self.update(get_defaults_dict(), _match_properties=False)
        return self

    checkinputs = param.Boolean(
        default=True,
        doc="""Check user input types, shapes at various stages and raise errors when they are not
within the designated constrains.""",
    )

    display = param.ClassSelector(
        Display,
        default=Display(),
        doc="""`Display` defaults-class containing display settings.
`(e.g. 'backend', 'animation', 'colorsequence', ...)`""",
    )


default_settings = DefaultConfig()
