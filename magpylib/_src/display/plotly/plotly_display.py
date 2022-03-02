""" plotly draw-functionalities"""
# pylint: disable=C0302
# pylint: disable=too-many-branches

import numbers
from itertools import cycle, combinations
from typing import Tuple
import warnings

try:
    import plotly.graph_objects as go
except ImportError as missing_module:  # pragma: no cover
    raise ModuleNotFoundError(
        """In order to use the plotly plotting backend, you need to install plotly via pip or conda,
        see https://github.com/plotly/plotly.py"""
    ) from missing_module

import numpy as np
from scipy.spatial.transform import Rotation as RotScipy
from magpylib import _src
from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.display.plotly.plotly_sensor_mesh import get_sensor_mesh
from magpylib._src.style import (
    get_style,
    LINESTYLES_MATPLOTLIB_TO_PLOTLY,
    SYMBOLS_MATPLOTLIB_TO_PLOTLY,
)
from magpylib._src.display.display_utility import (
    get_rot_pos_from_path,
    MagpyMarkers,
    draw_arrow_from_vertices,
    draw_arrowed_circle,
    place_and_orient_model3d,
)
from magpylib._src.defaults.defaults_utility import (
    SIZE_FACTORS_MATPLOTLIB_TO_PLOTLY,
    linearize_dict,
)

from magpylib._src.input_checks import check_excitations
from magpylib._src.utility import unit_prefix, format_obj_input
from magpylib._src.display.plotly.plotly_base_traces import (
    make_BaseCuboid,
    make_BaseCylinderSegment,
    make_BaseEllipsoid,
    make_BasePrism,
    # make_BaseCone,
    make_BaseArrow,
)
from magpylib._src.display.plotly.plotly_utility import (
    merge_mesh3d,
    merge_traces,
    getColorscale,
    getIntensity,
)


def make_Line(
    current=0.0,
    vertices=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly scatter3d parameters for a Line current in a dictionary based on the
    provided arguments
    """
    default_suffix = (
        f" ({unit_prefix(current)}A)"
        if current is not None
        else " (Current not initialized)"
    )
    name, name_suffix = get_name_and_suffix("Line", default_suffix, style)
    show_arrows = style.arrow.show
    arrow_size = style.arrow.size
    if show_arrows:
        vertices = draw_arrow_from_vertices(vertices, current, arrow_size)
    else:
        vertices = np.array(vertices).T
    if orientation is not None:
        vertices = orientation.apply(vertices.T).T
    x, y, z = (vertices.T + position).T
    line_width = style.arrow.width * SIZE_FACTORS_MATPLOTLIB_TO_PLOTLY["line_width"]
    line = dict(
        type="scatter3d",
        x=x,
        y=y,
        z=z,
        name=f"""{name}{name_suffix}""",
        mode="lines",
        line_width=line_width,
        line_color=color,
    )
    return {**line, **kwargs}


def make_Loop(
    current=0.0,
    diameter=1.0,
    position=(0.0, 0.0, 0.0),
    vert=50,
    orientation=None,
    color=None,
    style=None,
    **kwargs,
):
    """
    Creates the plotly scatter3d parameters for a Loop current in a dictionary based on the
    provided arguments
    """
    default_suffix = (
        f" ({unit_prefix(current)}A)"
        if current is not None
        else " (Current not initialized)"
    )
    name, name_suffix = get_name_and_suffix("Loop", default_suffix, style)
    arrow_size = style.arrow.size if style.arrow.show else 0
    vertices = draw_arrowed_circle(current, diameter, arrow_size, vert)
    if orientation is not None:
        vertices = orientation.apply(vertices.T).T
    x, y, z = (vertices.T + position).T
    line_width = style.arrow.width * SIZE_FACTORS_MATPLOTLIB_TO_PLOTLY["line_width"]
    circular = dict(
        type="scatter3d",
        x=x,
        y=y,
        z=z,
        name=f"""{name}{name_suffix}""",
        mode="lines",
        line_width=line_width,
        line_color=color,
    )
    return {**circular, **kwargs}


def make_DefaultTrace(
    obj, position=(0.0, 0.0, 0.0), orientation=None, color=None, style=None, **kwargs,
) -> dict:
    """
    Creates the plotly scatter3d parameters for an object with no specifically supported
    representation. The object will be reprensented by a scatter point and text above with object
    name.
    """

    default_suffix = ""
    name, name_suffix = get_name_and_suffix(
        f"{type(obj).__name__}", default_suffix, style
    )
    vertices = np.array([position])
    if orientation is not None:
        vertices = orientation.apply(vertices).T
    x, y, z = vertices
    trace = dict(
        type="scatter3d",
        x=x,
        y=y,
        z=z,
        name=f"""{name}{name_suffix}""",
        text=name,
        mode="markers+text",
        marker_size=10,
        marker_color=color,
        marker_symbol="diamond",
    )
    return {**trace, **kwargs}


def make_Dipole(
    moment=(0.0, 0.0, 1.0),
    position=(0.0, 0.0, 0.0),
    orientation=None,
    style=None,
    autosize=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Loop current in a dictionary based on the
    provided arguments
    """
    moment_mag = np.linalg.norm(moment)
    default_suffix = f" (moment={unit_prefix(moment_mag)}T/m³)"
    name, name_suffix = get_name_and_suffix("Dipole", default_suffix, style)
    size = style.size
    if autosize is not None:
        size *= autosize
    dipole = make_BaseArrow(
        base_vertices=10, diameter=0.3 * size, height=size, pivot=style.pivot
    )
    nvec = np.array(moment) / moment_mag
    zaxis = np.array([0, 0, 1])
    cross = np.cross(nvec, zaxis)
    dot = np.dot(nvec, zaxis)
    n = np.linalg.norm(cross)
    t = np.arccos(dot)
    vec = -t * cross / n if n != 0 else (0, 0, 0)
    mag_orient = RotScipy.from_rotvec(vec)
    orientation = orientation * mag_orient
    mag = np.array((0, 0, 1))
    return _update_mag_mesh(
        dipole, name, name_suffix, mag, orientation, position, style, **kwargs,
    )


def make_Cuboid(
    mag=(0.0, 0.0, 1000.0),
    dimension=(1.0, 1.0, 1.0),
    position=(0.0, 0.0, 0.0),
    orientation=None,
    style=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Cuboid Magnet in a dictionary based on the
    provided arguments
    """
    d = [unit_prefix(d / 1000) for d in dimension]
    default_suffix = f" ({d[0]}m|{d[1]}m|{d[2]}m)"
    name, name_suffix = get_name_and_suffix("Cuboid", default_suffix, style)
    cuboid = make_BaseCuboid(dimension=dimension)
    return _update_mag_mesh(
        cuboid, name, name_suffix, mag, orientation, position, style, **kwargs,
    )


def make_Cylinder(
    mag=(0.0, 0.0, 1000.0),
    base_vertices=50,
    diameter=1.0,
    height=1.0,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    style=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Cylinder Magnet in a dictionary based on the
    provided arguments
    """
    d = [unit_prefix(d / 1000) for d in (diameter, height)]
    default_suffix = f" (D={d[0]}m, H={d[1]}m)"
    name, name_suffix = get_name_and_suffix("Cylinder", default_suffix, style)
    cylinder = make_BasePrism(
        base_vertices=base_vertices, diameter=diameter, height=height,
    )
    return _update_mag_mesh(
        cylinder, name, name_suffix, mag, orientation, position, style, **kwargs,
    )


def make_CylinderSegment(
    mag=(0.0, 0.0, 1000.0),
    dimension=(1.0, 2.0, 1.0, 0.0, 90.0),
    position=(0.0, 0.0, 0.0),
    orientation=None,
    vert=25,
    style=None,
    **kwargs,
):
    """
    Creates the plotly mesh3d parameters for a Cylinder Segment Magnet in a dictionary based on the
    provided arguments
    """
    d = [unit_prefix(d / (1000 if i < 3 else 1)) for i, d in enumerate(dimension)]
    default_suffix = f" (r={d[0]}m|{d[1]}m, h={d[2]}m, φ={d[3]}°|{d[4]}°)"
    name, name_suffix = get_name_and_suffix("CylinderSegment", default_suffix, style)
    cylinder_segment = make_BaseCylinderSegment(*dimension, vert=vert)
    return _update_mag_mesh(
        cylinder_segment,
        name,
        name_suffix,
        mag,
        orientation,
        position,
        style,
        **kwargs,
    )


def make_Sphere(
    mag=(0.0, 0.0, 1000.0),
    vert=15,
    diameter=1,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    style=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Sphere Magnet in a dictionary based on the
    provided arguments
    """
    default_suffix = f" (D={unit_prefix(diameter / 1000)}m)"
    name, name_suffix = get_name_and_suffix("Sphere", default_suffix, style)
    vert = min(max(vert, 3), 20)
    sphere = make_BaseEllipsoid(vert=vert, dimension=[diameter] * 3)
    return _update_mag_mesh(
        sphere, name, name_suffix, mag, orientation, position, style, **kwargs,
    )


def make_Pixels(positions, size=1) -> dict:
    """
    Creates the plotly mesh3d parameters for Sensor pixels based on pixel positions and chosen size
    For now, only "cube" shape is provided.
    """
    pixels = [make_BaseCuboid(position=p, dimension=[size] * 3) for p in positions]
    return merge_mesh3d(*pixels)


def make_Sensor(
    pixel=(0.0, 0.0, 0.0),
    dimension=(1.0, 1.0, 1.0),
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    autosize=None,
    **kwargs,
):
    """
    Creates the plotly mesh3d parameters for a Sensor object in a dictionary based on the
    provided arguments

    size_pixels: float, default=1
        A positive number. Adjusts automatic display size of sensor pixels. When set to 0,
        pixels will be hidden, when greater than 0, pixels will occupy half the ratio of the minimum
        distance between any pixel of the same sensor, equal to `size_pixel`.
    """
    pixel = np.array(pixel)
    default_suffix = (
        f""" ({'x'.join(str(p) for p in pixel.shape[:-1])} pixels)"""
        if pixel.ndim != 1
        else ""
    )
    name, name_suffix = get_name_and_suffix("Sensor", default_suffix, style)
    style_arrows = style.arrows.as_dict(flatten=True, separator="_")
    sensor = get_sensor_mesh(**style_arrows, center_color=color)
    vertices = np.array([sensor[k] for k in "xyz"]).T
    if color is not None:
        sensor["facecolor"][sensor["facecolor"] == "rgb(238,238,238)"] = color
    dim = np.array(
        [dimension] * 3 if isinstance(dimension, (float, int)) else dimension[:3],
        dtype=float,
    )
    if autosize is not None:
        dim *= autosize
    if np.squeeze(pixel).ndim == 1:
        dim_ext = dim
    else:
        hull_dim = pixel.max(axis=0) - pixel.min(axis=0)
        dim_ext = max(np.mean(dim), np.min(hull_dim))
    cube_mask = (vertices < 1).all(axis=1)
    vertices[cube_mask] = 0 * vertices[cube_mask]
    vertices[~cube_mask] = dim_ext * vertices[~cube_mask]
    vertices /= 2  # sensor_mesh vertices are of length 2
    x, y, z = vertices.T
    sensor.update(x=x, y=y, z=z)
    meshes_to_merge = [sensor]
    if np.squeeze(pixel).ndim != 1:
        pixel_color = style.pixel.color
        pixel_size = style.pixel.size
        combs = np.array(list(combinations(pixel, 2)))
        vecs = np.diff(combs, axis=1)
        dists = np.linalg.norm(vecs, axis=2)
        pixel_dim = np.min(dists) / 2
        if pixel_size > 0:
            pixel_dim *= pixel_size
            pixels_mesh = make_Pixels(positions=pixel, size=pixel_dim)
            pixels_mesh["facecolor"] = np.repeat(pixel_color, len(pixels_mesh["i"]))
            meshes_to_merge.append(pixels_mesh)
        hull_pos = 0.5 * (pixel.max(axis=0) + pixel.min(axis=0))
        hull_dim[hull_dim == 0] = pixel_dim / 2
        hull_mesh = make_BaseCuboid(position=hull_pos, dimension=hull_dim)
        hull_mesh["facecolor"] = np.repeat(color, len(hull_mesh["i"]))
        meshes_to_merge.append(hull_mesh)
    sensor = merge_mesh3d(*meshes_to_merge)
    return _update_mag_mesh(
        sensor, name, name_suffix, orientation=orientation, position=position, **kwargs
    )


def _update_mag_mesh(
    mesh_dict,
    name,
    name_suffix,
    magnetization=None,
    orientation=None,
    position=None,
    style=None,
    **kwargs,
):
    """
    Updates an existing plotly mesh3d dictionary of an object which has a magnetic vector. The
    object gets colorized, positioned and oriented based on provided arguments
    """
    if hasattr(style, "magnetization"):
        color = style.magnetization.color
        if magnetization is not None and style.magnetization.show:
            vertices = np.array([mesh_dict[k] for k in "xyz"]).T
            color_middle = color.middle
            if color.mode == "tricycle":
                color_middle = kwargs.get("color", None)
            elif color.mode == "bicolor":
                color_middle = False
            mesh_dict["colorscale"] = getColorscale(
                color_transition=color.transition,
                color_north=color.north,
                color_middle=color_middle,
                color_south=color.south,
            )
            mesh_dict["intensity"] = getIntensity(
                vertices=vertices, axis=magnetization,
            )
    mesh_dict = place_and_orient_model3d(
        mesh_dict, orientation, position, showscale=False, name=f"{name}{name_suffix}",
    )
    return {**mesh_dict, **kwargs}


def get_name_and_suffix(default_name, default_suffix, style):
    """provides legend entry based on name and suffix"""
    name = default_name if style.label is None else style.label
    if style.description.show and style.description.text is None:
        name_suffix = default_suffix
    elif not style.description.show:
        name_suffix = ""
    else:
        name_suffix = f" ({style.description.text})"
    return name, name_suffix


def get_plotly_traces(
    input_obj,
    color=None,
    autosize=None,
    legendgroup=None,
    showlegend=None,
    legendtext=None,
    **kwargs,
) -> list:
    """
    This is a helper function providing the plotly traces for any object of the magpylib library. If
    the object is not supported, the trace representation will fall back to a single scatter point
    with the object name marked above it.

    - If the object has a path (multiple positions), the function will return both the object trace
    and the corresponding path trace. The legend entry of the path trace will be hidden but both
    traces will share the same `legendgroup` so that a legend entry click will hide/show both traces
    at once. From the user's perspective, the traces will be merged.

    - The argument caught by the kwargs dictionary must all be arguments supported both by
    `scatter3d` and `mesh3d` plotly objects, otherwise an error will be raised.
    """

    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-nested-blocks

    Sensor = _src.obj_classes.Sensor
    Cuboid = _src.obj_classes.Cuboid
    Cylinder = _src.obj_classes.Cylinder
    CylinderSegment = _src.obj_classes.CylinderSegment
    Sphere = _src.obj_classes.Sphere
    Dipole = _src.obj_classes.Dipole
    Loop = _src.obj_classes.Loop
    Line = _src.obj_classes.Line

    # parse kwargs into style and non style args
    style = get_style(input_obj, Config, **kwargs)
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("style")}
    kwargs["style"] = style
    style_color = getattr(style, "color", None)
    kwargs["color"] = style_color if style_color is not None else color
    kwargs["opacity"] = style.opacity
    legendgroup = f"{input_obj}" if legendgroup is None else legendgroup
    Config.display.context.colors[input_obj] = kwargs["color"]
    if hasattr(style, "magnetization"):
        if style.magnetization.show:
            check_excitations([input_obj])

    if hasattr(style, "arrow"):
        if style.arrow.show:
            check_excitations([input_obj])

    traces = []
    if isinstance(input_obj, MagpyMarkers):
        x, y, z = input_obj.markers.T
        marker = style.as_dict()["marker"]
        symb = marker["symbol"]
        marker["symbol"] = SYMBOLS_MATPLOTLIB_TO_PLOTLY.get(symb, symb)
        marker["size"] *= SIZE_FACTORS_MATPLOTLIB_TO_PLOTLY["marker_size"]
        default_name = "Marker" if len(x) == 1 else "Markers"
        default_suffix = "" if len(x) == 1 else f" ({len(x)} points)"
        name, name_suffix = get_name_and_suffix(default_name, default_suffix, style)
        trace = go.Scatter3d(
            name=f"{name}{name_suffix}",
            x=x,
            y=y,
            z=z,
            marker=marker,
            mode="markers",
            opacity=style.opacity,
        )
        traces.append(trace)
    else:
        if isinstance(input_obj, Sensor):
            kwargs.update(
                dimension=getattr(input_obj, "dimension", style.size),
                pixel=getattr(input_obj, "pixel", (0.0, 0.0, 0.0)),
                autosize=autosize,
            )
            make_func = make_Sensor
        elif isinstance(input_obj, Cuboid):
            kwargs.update(
                mag=input_obj.magnetization, dimension=input_obj.dimension,
            )
            make_func = make_Cuboid
        elif isinstance(input_obj, Cylinder):
            base_vertices = 50
            kwargs.update(
                mag=input_obj.magnetization,
                diameter=input_obj.dimension[0],
                height=input_obj.dimension[1],
                base_vertices=base_vertices,
            )
            make_func = make_Cylinder
        elif isinstance(input_obj, CylinderSegment):
            vert = 50
            kwargs.update(
                mag=input_obj.magnetization, dimension=input_obj.dimension, vert=vert,
            )
            make_func = make_CylinderSegment
        elif isinstance(input_obj, Sphere):
            kwargs.update(
                mag=input_obj.magnetization, diameter=input_obj.diameter,
            )
            make_func = make_Sphere
        elif isinstance(input_obj, Dipole):
            kwargs.update(
                moment=input_obj.moment, autosize=autosize,
            )
            make_func = make_Dipole
        elif isinstance(input_obj, Line):
            kwargs.update(
                vertices=input_obj.vertices, current=input_obj.current,
            )
            make_func = make_Line
        elif isinstance(input_obj, Loop):
            kwargs.update(
                diameter=input_obj.diameter, current=input_obj.current,
            )
            make_func = make_Loop
        elif getattr(input_obj, "children", None) is not None:
            make_func = None
        else:
            kwargs.update(obj=input_obj)
            make_func = make_DefaultTrace

        path_traces = []
        path_traces_extra = {}
        extra_model3d_traces = (
            style.model3d.data if style.model3d.data is not None else []
        )
        extra_model3d_traces = [
            t for t in extra_model3d_traces if t.backend == "plotly"
        ]
        for orient, pos in zip(*get_rot_pos_from_path(input_obj, style.path.frames)):
            if style.model3d.showdefault and make_func is not None:
                path_traces.append(
                    make_func(position=pos, orientation=orient, **kwargs)
                )
            for extr in extra_model3d_traces:
                if extr.show:
                    trace3d = {}
                    obj_extr_trace = (
                        extr.trace() if callable(extr.trace) else extr.trace
                    )
                    ttype = obj_extr_trace["type"]
                    if ttype == "mesh3d":
                        trace3d["showscale"] = False
                        if "facecolor" in obj_extr_trace:
                            ttype = "mesh3d_facecolor"
                    if ttype == "scatter3d":
                        trace3d["marker_color"] = kwargs["color"]
                        trace3d["line_color"] = kwargs["color"]
                    else:
                        trace3d["color"] = kwargs["color"]
                    trace3d.update(
                        linearize_dict(
                            place_and_orient_model3d(
                                obj_extr_trace,
                                orientation=orient,
                                position=pos,
                                scale=extr.scale,
                            ),
                            separator="_",
                        )
                    )
                    if ttype not in path_traces_extra:
                        path_traces_extra[ttype] = []
                    path_traces_extra[ttype].append(trace3d)
        trace = merge_traces(*path_traces)
        for ind, traces_extra in enumerate(path_traces_extra.values()):
            extra_model3d_trace = merge_traces(*traces_extra)
            label = (
                input_obj.style.label
                if input_obj.style.label is not None
                else str(type(input_obj).__name__)
            )
            extra_model3d_trace.update(
                {
                    "legendgroup": legendgroup,
                    "showlegend": showlegend and ind == 0 and not trace,
                    "name": label,
                }
            )
            traces.append(extra_model3d_trace)

        if trace:
            trace.update(
                {
                    "legendgroup": legendgroup,
                    "showlegend": True if showlegend is None else showlegend,
                }
            )
            if legendtext is not None:
                trace["name"] = legendtext
            traces.append(trace)

        if np.array(input_obj.position).ndim > 1 and style.path.show:
            scatter_path = make_path(input_obj, style, legendgroup, kwargs)
            traces.append(scatter_path)

    return traces


def make_path(input_obj, style, legendgroup, kwargs):
    """draw obj path based on path style properties"""
    x, y, z = np.array(input_obj.position).T
    txt_kwargs = (
        {"mode": "markers+text+lines", "text": list(range(len(x)))}
        if style.path.numbering
        else {"mode": "markers+lines"}
    )
    marker = style.path.marker.as_dict()
    symb = marker["symbol"]
    marker["symbol"] = SYMBOLS_MATPLOTLIB_TO_PLOTLY.get(symb, symb)
    marker["color"] = kwargs["color"] if marker["color"] is None else marker["color"]
    marker["size"] *= SIZE_FACTORS_MATPLOTLIB_TO_PLOTLY["marker_size"]
    line = style.path.line.as_dict()
    dash = line["style"]
    line["dash"] = LINESTYLES_MATPLOTLIB_TO_PLOTLY.get(dash, dash)
    line["color"] = kwargs["color"] if line["color"] is None else line["color"]
    line["width"] *= SIZE_FACTORS_MATPLOTLIB_TO_PLOTLY["line_width"]
    line = {k: v for k, v in line.items() if k != "style"}
    scatter_path = dict(
        type="scatter3d",
        x=x,
        y=y,
        z=z,
        name=f"Path: {input_obj}",
        showlegend=False,
        legendgroup=legendgroup,
        marker=marker,
        line=line,
        **txt_kwargs,
        opacity=kwargs["opacity"],
    )
    return scatter_path


def draw_frame(objs, color_sequence, zoom, autosize=None, **kwargs) -> Tuple:
    """
    Creates traces from input `objs` and provided parameters, updates the size of objects like
    Sensors and Dipoles in `kwargs` depending on the canvas size.

    Returns
    -------
    traces_dicts, kwargs: dict, dict
        returns the traces in a obj/traces_list dictionary and updated kwargs
    """
    return_autosize = False
    Sensor = _src.obj_classes.Sensor
    Dipole = _src.obj_classes.Dipole
    traces_dicts = {}
    # dipoles and sensors use autosize, the trace building has to be put at the back of the queue.
    # autosize is calculated from the other traces overall scene range
    traces_to_resize = {}
    for obj, color in zip(objs, cycle(color_sequence)):
        subobjs = [obj]
        legendgroup = None
        if getattr(obj, "children", None) is not None:
            subobjs.extend(obj.children)
            legendgroup = f"{obj}"
            if getattr(obj, "position", None) is not None:
                color = color if obj.style.color is None else obj.style.color
        first_shown = False
        for ind, subobj in enumerate(subobjs):
            if legendgroup is not None:
                if (
                    subobj.style.model3d.showdefault or ind + 1 == len(subobjs)
                ) and not first_shown:
                    # take name of parent
                    first_shown = True
                    if getattr(subobj, "children", None) is not None:
                        first_shown = any(m3.show for m3 in obj.style.model3d.data)
                    showlegend = True
                    legendtext = getattr(getattr(obj, "style", None), "label", None)
                    legendtext = f"{obj!r}" if legendtext is None else legendtext
                else:
                    legendtext = None
                    showlegend = False
                # print(f"{ind+1:02d}/{len(subobjs):02d} {legendtext=}, {showlegend=}")
            else:
                showlegend = True
                legendtext = None

            params = {
                **dict(
                    color=color,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    legendtext=legendtext,
                ),
                **kwargs,
            }
            if isinstance(subobj, (Dipole, Sensor)):
                traces_to_resize[subobj] = {**params}
                # temporary coordinates to be able to calculate ranges
                x, y, z = subobj.position.T
                traces_dicts[subobj] = [dict(x=x, y=y, z=z)]
            else:
                traces_dicts[subobj] = get_plotly_traces(subobj, **params)
    traces = [t for tr in traces_dicts.values() for t in tr]
    ranges = get_scene_ranges(*traces, zoom=zoom)
    if autosize is None or autosize == "return":
        if autosize == "return":
            return_autosize = True
        autosize = np.mean(np.diff(ranges)) / Config.display.autosizefactor
    for obj, params in traces_to_resize.items():
        traces_dicts[obj] = get_plotly_traces(obj, autosize=autosize, **params)
    if return_autosize:
        res = traces_dicts, autosize
    else:
        res = traces_dicts
    return res


def apply_fig_ranges(fig, ranges=None, zoom=None):
    """This is a helper function which applies the ranges properties of the provided `fig` object
    according to a certain zoom level. All three space direction will be equal and match the
    maximum of the ranges needed to display all objects, including their paths.

    Parameters
    ----------
    ranges: array of dimension=(3,2)
        min and max graph range

    zoom: float, default = 1
        When zoom=0 all objects are just inside the 3D-axes.

    Returns
    -------
    None: NoneType
    """
    if fig.data:
        if ranges is None:
            frames = fig.frames if fig.frames else [fig]
            traces = [t for frame in frames for t in frame.data]
            ranges = get_scene_ranges(*traces, zoom=zoom)
        scene_str = fig.data[-1].scene
        scene = getattr(fig.layout, "scene" if scene_str is None else scene_str)
        scene.update(
            **{
                f"{k}axis": dict(range=ranges[i], autorange=False, title=f"{k} [mm]")
                for i, k in enumerate("xyz")
            },
            aspectratio={k: 1 for k in "xyz"},
            aspectmode="manual",
            camera_eye={"x": 1, "y": -1.5, "z": 1.4},
        )


def get_scene_ranges(*traces, zoom=1) -> np.ndarray:
    """
    Returns 3x2 array of the min and max ranges in x,y,z directions of input traces. Traces can be
    any plotly trace object or a dict, with x,y,z numbered parameters.
    """
    if traces:
        ranges = {k: [] for k in "xyz"}
        for t in traces:
            for k, v in ranges.items():
                v.extend(
                    [
                        np.nanmin(np.array(t[k], dtype=float)),
                        np.nanmax(np.array(t[k], dtype=float)),
                    ]
                )
        r = np.array([[np.nanmin(v), np.nanmax(v)] for v in ranges.values()])
        size = np.diff(r, axis=1)
        size[size == 0] = 1
        m = size.max() / 2
        center = r.mean(axis=1)
        ranges = np.array([center - m * (1 + zoom), center + m * (1 + zoom)]).T
    else:
        ranges = np.array([[-1.0, 1.0]] * 3)
    return ranges


def animate_path(
    fig,
    objs,
    color_sequence=None,
    zoom=1,
    title="3D-Paths Animation",
    animation_time=3,
    animation_fps=30,
    animation_maxfps=50,
    animation_maxframes=200,
    animation_slider=False,
    animation_path=None,
    row=None,
    col=None,
    **kwargs,
):
    """This is a helper function which attaches plotly frames to the provided `fig` object
    according to a certain zoom level. All three space direction will be equal and match the
    maximum of the ranges needed to display all objects, including their paths.

    Parameters
    ----------
    animation_time: float, default = 3
        Sets the animation duration

    animation_fps: float, default = 30
        This sets the maximum allowed frame rate. In case of path positions needed to be displayed
        exceeds the `animation_fps` the path position will be downsampled to be lower or equal
        the `animation_fps`. This is mainly depending on the pc/browser performance and is set to
        50 by default to avoid hanging the animation process.

    animation_slider: bool, default = False
        if True, an interactive slider will be displayed and stay in sync with the animation

    title: str, default = "3D-Paths Animation"
        When zoom=0 all objects are just inside the 3D-axes.

    color_sequence: list or array_like, iterable, default=
            ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A',
            '#B68100', '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1',
            '#FC0080', '#B2828D', '#6C7C32', '#778AAE', '#862A16', '#A777F1',
            '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038']
        An iterable of color values used to cycle trough for every object displayed.
        A color and may be specified as:
      - A hex string (e.g. '#ff0000')
      - An rgb/rgba string (e.g. 'rgb(255,0,0)')
      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - A named CSS color

    Returns
    -------
    None: NoneTyp
    """
    if animation_path is None:
        path_indices, frame_duration, new_fps = get_animation_path_params(
            objs,
            animation_time=animation_time,
            animation_fps=animation_fps,
            animation_maxfps=animation_maxfps,
            animation_maxframes=animation_maxframes,
        )
    else:
        path_indices, frame_duration, new_fps = animation_path

    # calculate exponent of last frame index to avoid digit shift in
    # frame number display during animation
    exp = (
        np.log10(path_indices.max()).astype(int) + 1
        if path_indices.ndim != 0 and path_indices.max() > 0
        else 1
    )

    if animation_slider:
        sliders_dict = {
            "active": len(path_indices) - 1,
            "yanchor": "top",
            "font": {"size": 10},
            "xanchor": "left",
            "currentvalue": {
                "prefix": f"Fps={new_fps}, Path index: ",
                "visible": True,
                "xanchor": "right",
            },
            "pad": {"b": 10, "t": 10},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [],
        }

    buttons_dict = {
        "buttons": [
            {
                "args": [
                    None,
                    {
                        "frame": {"duration": frame_duration},
                        "transition": {"duration": 0},
                        "fromcurrent": True,
                    },
                ],
                "label": "Play",
                "method": "animate",
            },
            {
                "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                "label": "Pause",
                "method": "animate",
            },
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 20},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top",
    }

    # create frame for each path index or downsampled path index
    frames = []
    autosize = "return"
    for i, ind in enumerate(path_indices):
        kwargs["style_path_frames"] = [ind]
        frame = draw_frame(objs, color_sequence, zoom, autosize=autosize, **kwargs,)
        if i == 0:  # get the dipoles and sensors autosize from first frame
            traces_dicts, autosize = frame
        else:
            traces_dicts = frame
        traces = [t for tr in traces_dicts.values() for t in tr]
        frames.append(
            dict(
                data=traces,
                name=str(ind + 1),
                layout=dict(title=f"""{title} - path index: {ind+1:0{exp}d}"""),
            )
        )
        if animation_slider:
            slider_step = {
                "args": [
                    [str(ind + 1)],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate",},
                ],
                "label": str(ind + 1),
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

    # update fig
    fig.add_traces(frames[0]["data"], rows=row, cols=col)
    for f in frames:
        for t in f["data"]:
            t["scene"] = fig.data[-1].scene
    fig.frames = frames
    clean_legendgroups(fig)
    fig.update_layout(
        # height=None,
        title=title,
        updatemenus=[buttons_dict],
        sliders=[sliders_dict] if animation_slider else None,
    )
    apply_fig_ranges(fig, zoom=zoom)


def get_animation_path_params(
    objs,
    animation_time=3,
    animation_fps=30,
    animation_maxfps=50,
    animation_maxframes=200,
    animation_slider=False,
):
    """Make sure the number of frames does not exceed the max frames and max frame rate
    # downsample if necessary"""
    path_lengths = []
    for obj in objs:
        subobjs = [obj]
        if getattr(obj, "_object_type", None) == "Collection":
            subobjs.extend(obj.children)
        for subobj in subobjs:
            path_len = getattr(subobj, "_position", np.array((0.0, 0.0, 0.0))).shape[0]
            path_lengths.append(path_len)

    max_pl = max(path_lengths)
    if animation_fps > animation_maxfps:
        warnings.warn(
            f"The set `animation_fps` at {animation_fps} is greater than the max allowed of"
            f" {animation_maxfps}. `animation_fps` will be set to {animation_maxfps}. "
            f"You can modify the default value by setting it in "
            "`magpylib.defaults.display.animation.maxfps`"
        )
        animation_fps = animation_maxfps

    maxpos = min(animation_time * animation_fps, animation_maxframes)

    if max_pl <= maxpos:
        path_indices = np.arange(max_pl)
    else:
        round_step = max_pl / (maxpos - 1)
        ar = np.linspace(0, max_pl, max_pl, endpoint=False)
        path_indices = np.unique(np.floor(ar / round_step) * round_step).astype(
            int
        )  # downsampled indices
        path_indices[-1] = (
            max_pl - 1
        )  # make sure the last frame is the last path position

    frame_duration = int(animation_time * 1000 / path_indices.shape[0])
    new_fps = int(1000 / frame_duration)
    if max_pl > animation_maxframes:
        warnings.warn(
            f"The number of frames ({max_pl}) is greater than the max allowed "
            f"of {animation_maxframes}. The `animation_fps` will be set to {new_fps}. "
            f"You can modify the default value by setting it in "
            "`magpylib.defaults.display.animation.maxframes`"
        )

    return path_indices, frame_duration, new_fps


def display_plotly(
    *obj_list,
    markers=None,
    zoom=1,
    fig=None,
    renderer=None,
    animation=False,
    color_sequence=None,
    row=None,
    col=None,
    animation_path=None,
    **kwargs,
):

    """
    Display objects and paths graphically using the plotly library.

    Parameters
    ----------
    objects: sources, collections or sensors
        Objects to be displayed.

    markers: array_like, None, shape (N,3), default=None
        Display position markers in the global CS. By default no marker is displayed.

    zoom: float, default = 1
        Adjust plot zoom-level. When zoom=0 all objects are just inside the 3D-axes.

    fig: plotly Figure, default=None
        Display graphical output in a given figure:
        - plotly.graph_objects.Figure
        - plotly.graph_objects.FigureWidget
        By default a new `Figure` is created and displayed.

    renderer: str. default=None,
        The renderers framework is a flexible approach for displaying plotly.py figures in a variety
        of contexts.
        Available renderers are:
        ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
         'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
         'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
         'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
         'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']

    title: str, default = "3D-Paths Animation"
        When zoom=0 all objects are just inside the 3D-axes.

    color_sequence: list or array_like, iterable, default=
            ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A',
            '#B68100', '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1',
            '#FC0080', '#B2828D', '#6C7C32', '#778AAE', '#862A16', '#A777F1',
            '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038']
        An iterable of color values used to cycle trough for every object displayed.
        A color and may be specified as:
      - A hex string (e.g. '#ff0000')
      - An rgb/rgba string (e.g. 'rgb(255,0,0)')
      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - A named CSS color

    Returns
    -------
    None: NoneType
    """

    flat_obj_list = format_obj_input(obj_list)

    show_fig = False
    if fig is None:
        show_fig = True
        fig = go.Figure()

    # Check animation parameters
    animation, kwargs, _ = process_animation_kwargs(animation, kwargs, flat_obj_list)

    if obj_list:
        style = getattr(obj_list[0], "style", None)
        label = getattr(style, "label", None)
        title = label if len(obj_list) == 1 else None
    else:
        title = "No objects to be displayed"

    if markers is not None and markers:
        obj_list = list(obj_list) + [MagpyMarkers(*markers)]

    if color_sequence is None:
        color_sequence = Config.display.colorsequence
    with fig.batch_update():
        if animation is not False:
            animated_subplots = Config.display.context.subplots
            if (row is not None or col is not None) and not animated_subplots:
                raise NotImplementedError(
                    "Animation in combination with subplots must be called via the `with` "
                    "statement. Check the `magpylib.display_context` docstring for examples."
                )
            if kwargs.get("field", "").startswith(("H", "B")):
                return draw_sensor_values(
                    flat_obj_list,
                    fig=fig,
                    row=row,
                    col=col,
                    animation_path=animation_path,
                    **kwargs,
                )
            title = "3D-Paths Animation" if title is None else title
            animate_path(
                fig=fig,
                objs=obj_list,
                color_sequence=color_sequence,
                zoom=zoom,
                title=title,
                row=row,
                col=col,
                animation_path=animation_path,
                **kwargs,
            )
        else:
            traces_dicts = draw_frame(obj_list, color_sequence, zoom, **kwargs)
            traces = [t for traces in traces_dicts.values() for t in traces]
            fig.add_traces(traces, rows=row, cols=col)
            clean_legendgroups(fig)
            fig.update_layout(title_text=title)
            apply_fig_ranges(fig, zoom=zoom)
        fig.update_layout(legend_itemsizing="constant")
    if show_fig:
        fig.show(renderer=renderer)


def process_animation_kwargs(animation, kwargs, flat_obj_list=None):
    """extracts animation kwargs"""
    if (
        isinstance(animation, numbers.Number)
        and not isinstance(animation, bool)
        and animation > 0
    ):
        kwargs["animation_time"] = animation
        animation = True
    if flat_obj_list is not None:
        if (
            not any(
                getattr(obj, "position", np.array([])).ndim > 1 for obj in flat_obj_list
            )
            and animation is not False
        ):  # check if some path exist for any object
            animation = False
            warnings.warn("No path to be animated detected, displaying standard plot")

    no_anim_kwargs = {}
    animation_kwargs = {}
    for k, v in kwargs.items():
        if k.split("_")[0] == "animation":
            if k != "animation":
                animation_kwargs[k] = v
        else:
            no_anim_kwargs[k] = v

    if animation is False:
        kwargs = no_anim_kwargs
    else:
        disp_props = Config.display.animation.copy()
        disp_props.update(
            {k[len("animation") + 1 :]: v for k, v in animation_kwargs.items()}
        )
        animation_kwargs = disp_props.as_dict(flatten=True, separator="_")
        animation_kwargs = {f"animation_{k}": v for k, v in animation_kwargs.items()}
        kwargs = {**no_anim_kwargs, **animation_kwargs}
    return animation, kwargs, animation_kwargs


def clean_legendgroups(fig):
    """removes legend duplicates"""
    frames = [fig.data]
    if fig.frames:
        data_list = [f["data"] for f in fig.frames]
        frames.extend(data_list)
    for f in frames:
        legendgroups = []
        for t in f:
            if t.legendgroup not in legendgroups:
                legendgroups.append(t.legendgroup)
            else:
                t.showlegend = False


def batch_animate_subplots(ctx, show_fn, **kwargs):
    """draw objects into canvas in a batch"""
    all_objs = [plot["objects"] for plot in ctx.subplots]
    all_objs = format_obj_input(all_objs, allow="sources+sensors+collections")

    anim_kwargs = process_animation_kwargs(kwargs.get("animation", True), kwargs)[-1]
    anim_path = get_animation_path_params(all_objs, **anim_kwargs)

    fig = ctx.canvas
    with fig.batch_update():
        for ind, plot in enumerate(ctx.subplots):
            subfig = plot["kwargs"]["canvas"]
            show_fn(*plot["objects"], **plot["kwargs"], animation_path=anim_path)
            fig.add_traces(subfig.data)
            if ind == 0:
                frames = subfig.frames
            else:
                for f1, f2 in zip(frames, subfig.frames):
                    data = list(f1["data"])
                    data.extend(list(f2["data"]))
                    f1["data"] = data
            fig.update_layout(subfig.layout)
            if hasattr(subfig.data[-1], "scene"):
                scene_str = subfig.data[-1].scene
                getattr(fig.layout, scene_str).update(getattr(subfig.layout, scene_str))
        fig.frames = frames
        clean_legendgroups(fig)


def draw_sensor_values(
    flat_obj_list, fig, row, col, animation_path, field="B", layout=None, **kwargs
):
    """draws and animates sensor values over a path in a subplot"""
    if layout is None:
        layout = {}
    layout_kwargs = layout
    layout_kwargs.update({k[len('layout_'):]:v for k,v in kwargs.items() if k.startswith('layout_')})
    print(layout_kwargs)
    sources = format_obj_input(flat_obj_list, allow="sources")
    sensors = format_obj_input(flat_obj_list, allow="sensors")
    # pylint: disable=import-outside-toplevel
    from magpylib._src.fields.field_wrap_BH_level3 import getBH_level2

    frames_indices = np.array(animation_path[0])
    coords_indices = {0, 1, 2}
    bh = field
    if len(field) > 1:
        coords_indices = set("xyz".index(k) for k in field[1:])
        bh = field[0]
    xyz_linestyles = ("solid", "dash", "dot")

    BH_array = getBH_level2(sources, sensors, sumup=True, squeeze=False, field=bh)
    BH_array = BH_array[0]  # select first source
    BH_array = BH_array.swapaxes(0, 1)  # swap axes to have sensors first, path second
    if BH_array.ndim == 4:  # average on pixel if any
        BH_array = BH_array.mean(axis=-2)

    for sens, BH in zip(sensors, BH_array):
        color = Config.display.context.colors.get(sens, None)
        for i in coords_indices:
            k = "xyz"[i]
            kwargs = dict(
                name=f"{field}{k}_{sens}",
                legendgroup=f"{sens}",
                showlegend=False,
                row=row,
                col=col,
            )
            fig.add_scatter(
                x=frames_indices,
                y=BH.T[i][frames_indices],
                mode="lines",
                line_dash=xyz_linestyles[i],
                line_color=color,
                **kwargs,
            )
            fig.add_scatter(
                x=[frames_indices[-1]],
                y=[BH.T[i][frames_indices[-1]]],
                mode="markers",
                marker_size=10,
                marker_color=color,
                **kwargs,
            )
        t = fig.data[-1]
        xaxis, yaxis = t.xaxis, t.yaxis
        m, M = min(frames_indices), max(frames_indices)
        fig_xaxis = getattr(
            fig.layout, "xaxis" if xaxis in (None, "x") else "xaxis" + xaxis[1]
        )
        fig_xaxis.range = [
            m - (M - m) * 0.05,
            M + (M - m) * 0.05,
        ]
        # fig_xaxis.title = 'Path indices'
        m, M = np.min(BH_array), np.max(BH_array)
        fig_yaxis = getattr(
            fig.layout, "yaxis" if yaxis in (None, "y") else "yaxis" + yaxis[1]
        )
        fig_yaxis.title = f"{field} [mT]"
        fig_yaxis.range = [
            m - (M - m) * 0.05,
            M + (M - m) * 0.05,
        ]
        # fig_yaxis.ticklabelposition="inside top"
    frames = []
    for ind, _ in enumerate(frames_indices):
        data = []
        for sens, BH in zip(sensors, BH_array):
            color = Config.display.context.colors.get(sens, None)
            for i in coords_indices:
                k = "xyz"[i]
                kwargs = dict(
                    name=f"{field}_{sens}",
                    xaxis=xaxis,
                    yaxis=yaxis,
                    legendgroup=f"{sens}",
                    showlegend=False,
                )
                data.extend(
                    [
                        go.Scatter(
                            x=frames_indices,
                            y=BH.T[i][frames_indices],
                            mode="lines",
                            line_dash=xyz_linestyles[i],
                            line_color=color,
                            **kwargs,
                        ),
                        go.Scatter(
                            x=[frames_indices[ind]],
                            y=[BH.T[i][frames_indices[ind]]],
                            mode="markers",
                            marker_size=10,
                            marker_color=color,
                            **kwargs,
                        ),
                    ]
                )
        frames.append(dict(data=data))
    fig.frames = frames
    fig.update_layout(**layout_kwargs)
