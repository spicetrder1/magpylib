""" Display function codes"""

import warnings
from contextlib import contextmanager
from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.utility import format_obj_input, test_path_format
from magpylib._src.display.display_matplotlib import display_matplotlib
from magpylib._src.input_checks import (
    check_dimensions,
    check_excitations,
    check_format_input_backend,
    check_input_zoom,
    check_input_animation,
    check_format_input_vector,
)


# ON INTERFACE
def _show(
    *objects,
    zoom=0,
    animation=False,
    markers=None,
    backend=None,
    canvas=None,
    **kwargs,
):
    """
    Display objects and paths graphically.

    The private function is needed to intercept `show` kwargs from the `display_context` manager.

    See `show` function for extended docstring
    """
    # flatten input
    obj_list_flat = format_obj_input(objects, allow="sources+sensors")
    obj_list_semi_flat = format_obj_input(objects, allow="sources+sensors+collections")

    # test if all source dimensions and excitations have been initialized
    check_dimensions(obj_list_flat)
    check_excitations(obj_list_flat)

    # test if every individual obj_path is good
    test_path_format(obj_list_flat)

    # input checks
    backend = check_format_input_backend(backend)
    check_input_zoom(zoom)
    check_input_animation(animation)
    check_format_input_vector(
        markers,
        dims=(2,),
        shape_m1=3,
        sig_name="markers",
        sig_type="array_like of shape (n,3)",
        allow_None=True,
    )

    if backend == "matplotlib":
        if animation is not False:
            msg = "The matplotlib backend does not support animation at the moment.\n"
            msg += "Use `backend=plotly` instead."
            warnings.warn(msg)
            # animation = False
        display_matplotlib(
            *obj_list_semi_flat, markers=markers, zoom=zoom, canvas=canvas, **kwargs,
        )
    elif backend == "plotly":
        # pylint: disable=import-outside-toplevel
        from magpylib._src.display.plotly.plotly_display import display_plotly

        display_plotly(
            *obj_list_semi_flat,
            markers=markers,
            zoom=zoom,
            fig=canvas,
            animation=animation,
            **kwargs,
        )


def show(
    *objects,
    zoom=0,
    animation=False,
    markers=None,
    backend=None,
    canvas=None,
    **kwargs,
):
    """Display objects and paths graphically.

    Global graphic styles can be set with kwargs as style-dictionary or using
    style-underscore_magic.

    Parameters
    ----------
    objects: Magpylib objects (sources, collections, sensors)
        Objects to be displayed.

    zoom: float, default=`0`
        Adjust plot zoom-level. When zoom=0 3D-figure boundaries are tight.

    animation: bool or float, default=`False`
        If `True` and at least one object has a path, the paths are rendered.
        If input is a positive float, the animation time is set to the given value.
        This feature is only available for the plotly backend.

    markers: array_like, shape (n,3), default=`None`
        Display position markers in the global coordinate system.

    backend: string, default=`None`
        Define plotting backend. Must be one of `'matplotlib'` or `'plotly'`. If not
        set, parameter will default to `magpylib.defaults.display.backend` which is
        `'matplotlib'` by installation default.

    canvas: matplotlib.pyplot `AxesSubplot` or plotly `Figure` object, default=`None`
        Display graphical output on a given canvas:
        - with matplotlib: `matplotlib.axes._subplots.AxesSubplot` with `projection=3d.
        - with plotly: `plotly.graph_objects.Figure` or `plotly.graph_objects.FigureWidget`.
        By default a new canvas is created and immediately displayed.

    Returns
    -------
    `None`: NoneType

    Examples
    --------

    Display multiple objects, object paths, markers in 3D using Matplotlib or Plotly:

    >>> import magpylib as magpy
    >>> src = magpy.magnet.Sphere(magnetization=(0,0,1), diameter=1)
    >>> src.move([(0.1*x,0,0) for x in range(50)])
    >>> src.rotate_from_angax(angle=range(0,400,10), axis='z', anchor=0, start=11)
    >>> ts = [-.4,0,.4]
    >>> sens = magpy.Sensor(position=(0,0,2), pixel=[(x,y,0) for x in ts for y in ts])
    >>> magpy.show(src, sens)
    >>> magpy.show(src, sens, backend='plotly')
    --> graphic output

    Display output on your own canvas (here a Matplotlib 3d-axes):

    >>> import matplotlib.pyplot as plt
    >>> import magpylib as magpy
    >>> my_axis = plt.axes(projection='3d')
    >>> magnet = magpy.magnet.Cuboid(magnetization=(1,1,1), dimension=(1,2,3))
    >>> sens = magpy.Sensor(position=(0,0,3))
    >>> magpy.show(magnet, sens, canvas=my_axis, zoom=1)
    >>> plt.show()
    --> graphic output

    Use sophisticated figure styling options accessible from defaults, as individual object styles
    or as global style arguments in display.

    >>> import magpylib as magpy
    >>> src1 = magpy.magnet.Sphere((1,1,1), 1, [(0,0,0), (0,0,3)])
    >>> src2 = magpy.magnet.Sphere((1,1,1), 1, [(1,0,0), (1,0,3)], style_path_show=False)
    >>> magpy.defaults.display.style.magnet.magnetization.size = 2
    >>> src1.style.magnetization.size = 1
    >>> magpy.show(src1, src2, style_color='r')
    --> graphic output
    """
    kwargs = {**getattr(Config.display, "_kwargs", {}), **kwargs}
    # TODO find a better way to override within `with display_context` only values that are
    # different from the `show` function signature defaults
    # Example:
    # with magpy.display_context(canvas=fig, zoom=1):
    #   src1.show(row=1, col=1)
    #   magpy.show(src2, row=1, col=2)
    #   magpy.show(src1, src2, row=1, col=3, zoom=10)
    # # -> zoom=10 should override zoom=1 from context

    input_kwargs = dict(
        zoom=zoom, animation=animation, markers=markers, backend=backend, canvas=canvas,
    )
    defaults_kwargs = dict(
        zoom=0, animation=False, markers=None, backend=None, canvas=None,
    )
    for k, v in input_kwargs.items():
        if v != defaults_kwargs[k]:
            kwargs[k] = v
    _show(*objects, **kwargs)


@contextmanager
def display_context(**kwargs):
    """Context manager to temporarily set display settings in the `with` statement context.

    You need to invoke as ``display_context(pattern1=value1, pattern2=value2)``.

    Examples
    --------
    >>> import magpylib as magpy
    >>> magpy.defaults.reset() # may be necessary in a live kernel context
    >>> cube = magpy.magnet.Cuboid((0,0,1),(1,1,1))
    >>> cylinder = magpy.magnet.Cylinder((0,0,1),(1,1))
    >>> sphere = magpy.magnet.Sphere((0,0,1),diameter=1)
    >>> with magpy.display_context(backend='plotly'):
    >>>     cube.show() # -> displays with plotly
    >>>     cylinder.show() # -> displays with plotly
    >>> sphere.show() # -> displays with matplotlib
    """
    # pylint: disable=protected-access
    if not hasattr(Config.display, "_kwargs"):
        Config.display._kwargs = {}
    conf_disp_orig = {**Config.display._kwargs}
    try:
        Config.display._kwargs.update(**kwargs)
        yield _show
    finally:
        Config.display._kwargs = {**conf_disp_orig}
