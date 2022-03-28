"""base traces building functions"""
from functools import partial
import numpy as np
from magpylib._src.display.plotly.plotly_utility import merge_mesh3d
from magpylib._src.display.display_utility import place_and_orient_model3d


def base_validator(name, value, conditions):
    """Validates value based on dictionary of conditions"""

    msg = f"""Input {name} must be one of `{tuple(conditions.keys())},`
received {value!r} instead.
"""
    assert value in conditions.keys(), msg
    return conditions[value]


validate_pivot = partial(base_validator, "pivot")


def get_model(trace, *, backend, show, scale, kwargs):
    """Returns model3d dict depending on backend"""

    model = dict(constructor="Mesh3d", kwargs=trace, args=(), show=show, scale=scale)
    if backend == "matplotlib":
        x, y, z, i, j, k = [trace[k] for k in "xyzijk"]
        triangles = np.array([i, j, k]).T
        model.update(
            constructor="plot_trisurf", args=(x, y, z), kwargs={"triangles": triangles}
        )
    model["kwargs"].update(kwargs)
    if backend == "plotly-dict":
        model = {"type": "mesh3d", **model["kwargs"]}
    else:
        model["backend"] = backend
        model["kwargs"].pop("type", None)
    return model


def make_Cuboid(
    backend,
    dimension=(1.0, 1.0, 1.0),
    position=None,
    orientation=None,
    show=True,
    scale=1,
    **kwargs,
) -> dict:
    """Provides the 3D-model parameters for a cuboid in dictionary form, based on
    the given dimension. The zero position is in the barycenter of the vertices.

    Parameters
    ----------
    dimension : 3-tuple, default=(1,1,1)
        Length of the cuboid sides `x,y,z`.

    position: array_like, shape (3,), default=(0,0,0)
        Reference position of the vertices in the global CS. The zero position is
        in the barycenter of the vertices.

    orientation: scipy `Rotation` object with length 1 or m, default=`identity`
        Orientation of the vertices in the global CS.

    Returns
    -------
    3D-model: dict
        A dictionary with necessary key/value pairs with the necessary information to construct
        a 3D-model.
    """
    dimension = np.array(dimension, dtype=float)
    trace = dict(
        i=np.array([7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7]),
        j=np.array([0, 7, 1, 2, 6, 7, 1, 2, 5, 5, 2, 2]),
        k=np.array([3, 4, 2, 3, 5, 6, 5, 5, 0, 1, 7, 6]),
        x=np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * 0.5 * dimension[0],
        y=np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * 0.5 * dimension[1],
        z=np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * 0.5 * dimension[2],
    )

    trace = place_and_orient_model3d(trace, orientation=orientation, position=position)
    return get_model(trace, backend=backend, show=show, scale=scale, kwargs=kwargs)


def make_Prism(
    backend,
    base=3,
    diameter=1.0,
    height=1.0,
    position=None,
    orientation=None,
    show=True,
    scale=1,
    **kwargs,
) -> dict:
    """Provides the 3D-model parameters for a prism in dictionary form, based on
    number of vertices of the base, diameter and height. The zero position is in the
    barycenter of the vertices.

    Parameters
    ----------
    base : int, default=6
        Number of vertices of the base in the xy-plane.

    diameter : float, default=1
        Diameter dimension inscribing the base.

    height : float, default=1
        Prism height in the z-direction.

    position: array_like, shape (3,), default=(0,0,0)
        Reference position of the vertices in the global CS. The zero position is
        in the barycenter of the vertices.

    orientation: scipy Rotation object with length 1 or m, default=`identity`
        Orientation of the vertices in the global CS.

    Returns
    -------
    3D-model: dict
        A dictionary with necessary key/value pairs with the necessary information to construct
        a 3D-model.
    """
    N = base
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    c1 = np.array([1 * np.cos(t), 1 * np.sin(t), t * 0 - 1]) * 0.5
    c2 = np.array([1 * np.cos(t), 1 * np.sin(t), t * 0 + 1]) * 0.5
    c3 = np.array([[0, 0], [0, 0], [-1, 1]]) * 0.5
    c = np.concatenate([c1, c2, c3], axis=1)
    c = c.T * np.array([diameter, diameter, height])
    i1 = np.arange(N)
    j1 = i1 + 1
    j1[-1] = 0
    k1 = i1 + N

    i2 = i1 + N
    j2 = j1 + N
    j2[-1] = N
    k2 = i1 + 1
    k2[-1] = 0

    i3 = i1
    j3 = j1
    k3 = i1 * 0 + 2 * N

    i4 = i2
    j4 = j2
    k4 = k3 + 1

    # k2&j2 and k3&j3 inverted because of face orientation
    i = np.concatenate([i1, i2, i3, i4])
    j = np.concatenate([j1, k2, k3, j4])
    k = np.concatenate([k1, j2, j3, k4])

    x, y, z = c.T
    trace = dict(x=x, y=y, z=z, i=i, j=j, k=k)
    trace = place_and_orient_model3d(trace, orientation=orientation, position=position)
    return get_model(trace, backend=backend, show=show, scale=scale, kwargs=kwargs)


def make_Ellipsoid(
    backend,
    dimension=(1.0, 1.0, 1.0),
    vert=15,
    position=None,
    orientation=None,
    show=True,
    scale=1,
    **kwargs,
) -> dict:
    """Provides the 3D-model parameters for an ellipsoid in dictionary form, based
    on number of vertices of the circumference, and the dimension. The zero position is in the
    barycenter of the vertices.

    Parameters
    ----------
    dimension : tuple, default=(1.0, 1.0, 1.0)
        Dimension in the `x,y,z` directions.

    vert : int, default=15
        Number of vertices along the circumference.

    position: array_like, shape (3,), default=(0,0,0)
        Reference position of the vertices in the global CS. The zero position is
        in the barycenter of the vertices.

    orientation: scipy Rotation object with length 1 or m, default=`identity`
        Orientation of the vertices in the global CS.

    Returns
    -------
    3D-model: dict
        A dictionary with necessary key/value pairs with the necessary information to construct
        a 3D-model.
    """
    N = vert
    phi = np.linspace(0, 2 * np.pi, vert, endpoint=False)
    theta = np.linspace(-np.pi / 2, np.pi / 2, vert, endpoint=True)
    phi, theta = np.meshgrid(phi, theta)

    x = np.cos(theta) * np.sin(phi) * dimension[0] * 0.5
    y = np.cos(theta) * np.cos(phi) * dimension[1] * 0.5
    z = np.sin(theta) * dimension[2] * 0.5

    x, y, z = x.flatten()[N - 1 :], y.flatten()[N - 1 :], z.flatten()[N - 1 :]

    i1 = [0] * N
    j1 = np.array([N] + list(range(1, N)), dtype=int)
    k1 = np.array(list(range(1, N)) + [N], dtype=int)

    i2 = np.concatenate([k1 + i * N for i in range(N - 2)])
    j2 = np.concatenate([j1 + i * N for i in range(N - 2)])
    k2 = np.concatenate([j1 + (i + 1) * N for i in range(N - 2)])

    i3 = np.concatenate([k1 + i * N for i in range(N - 2)])
    j3 = np.concatenate([j1 + (i + 1) * N for i in range(N - 2)])
    k3 = np.concatenate([k1 + (i + 1) * N for i in range(N - 2)])

    i = np.concatenate([i1, i2, i3])
    j = np.concatenate([j1, j2, j3])
    k = np.concatenate([k1, k2, k3])

    trace = dict(x=x, y=y, z=z, i=i, j=j, k=k)
    trace = place_and_orient_model3d(trace, orientation=orientation, position=position)
    return get_model(trace, backend=backend, show=show, scale=scale, kwargs=kwargs)


def make_CylinderSegment(
    backend,
    dimension=(1.0, 2.0, 1.0, 0.0, 90.0),
    vert=50,
    position=None,
    orientation=None,
    show=True,
    scale=1,
    **kwargs,
) -> dict:
    """Provides the 3D-model parameters for a cylinder segment in dictionary form, based on
    inner and outer diameters, height, and section angles in degrees. The zero position is at
    `z=0` at the center point of the arcs.

    Parameters
    ----------
    dimension: array_like, shape (5,), default=`None`
        Dimension/Size of the cylinder segment of the form (r1, r2, h, phi1, phi2)
        where r1<r2 denote inner and outer radii in units of [mm], phi1<phi2 denote
        the cylinder section angles in units of [deg] and h is the cylinder height
        in units of [mm].

    vert : int, default=50
        Number of vertices along a the complete 360 degrees arc. The number along the phi1-phi2-arc
        is computed with `max(5, int(vert * abs(phi1 - phi2) / 360))`.

    position: array_like, shape (3,), default=(0,0,0)
        Reference position of the vertices in the global CS. The zero position is
        in the barycenter of the vertices.

    orientation: scipy Rotation object with length 1 or m, default=`identity`
        Orientation of the vertices in the global CS.

    Returns
    -------
    3D-model: dict
        A dictionary with necessary key/value pairs with the necessary information to construct
        a 3D-model.
    """
    r1, r2, h, phi1, phi2 = dimension
    N = max(5, int(vert * abs(phi1 - phi2) / 360))
    phi = np.linspace(phi1, phi2, N)
    x = np.cos(np.deg2rad(phi))
    y = np.sin(np.deg2rad(phi))
    z = np.zeros(N)
    c1 = np.array([r1 * x, r1 * y, z + h / 2])
    c2 = np.array([r2 * x, r2 * y, z + h / 2])
    c3 = np.array([r1 * x, r1 * y, z - h / 2])
    c4 = np.array([r2 * x, r2 * y, z - h / 2])
    x, y, z = np.concatenate([c1, c2, c3, c4], axis=1)

    i1 = np.arange(N - 1)
    j1 = i1 + N
    k1 = i1 + 1

    i2 = k1
    j2 = j1
    k2 = j1 + 1

    i3 = i1
    j3 = k1
    k3 = j1 + N

    i4 = k3 + 1
    j4 = k3
    k4 = k1

    i5 = np.array([0, N])
    j5 = np.array([2 * N, 0])
    k5 = np.array([3 * N, 3 * N])

    i = [i1, i2, i1 + 2 * N, i2 + 2 * N, i3, i4, i3 + N, i4 + N]
    j = [j1, j2, k1 + 2 * N, k2 + 2 * N, j3, j4, k3 + N, k4 + N]
    k = [k1, k2, j1 + 2 * N, j2 + 2 * N, k3, k4, j3 + N, j4 + N]

    if phi2 - phi1 != 360:
        i.extend([i5, i5 + N - 1])
        j.extend([k5, k5 + N - 1])
        k.extend([j5, j5 + N - 1])
    i, j, k = (np.hstack(l) for l in (i, j, k))

    trace = dict(x=x, y=y, z=z, i=i, j=j, k=k)
    trace = place_and_orient_model3d(trace, orientation=orientation, position=position)
    return get_model(trace, backend=backend, show=show, scale=scale, kwargs=kwargs)


def make_Pyramid(
    backend,
    base=3,
    diameter=1,
    height=1,
    pivot="middle",
    position=None,
    orientation=None,
    show=True,
    scale=1,
    **kwargs,
) -> dict:
    """Provides the 3D-model parameters for a pyramid in dictionary form, based on
    number of vertices of the base, diameter and height. The zero position is in the
    barycenter of the vertices.

    Parameters
    ----------
    base : int, default=30
        Number of vertices of the cone base.

    diameter : float, default=1
        Diameter of the cone base.

    height : int, default=1
        Pyramid height.

    pivot : str, default='middle'
        The part of the cone that is anchored to the grid and about which it rotates.
        Can be one of `['tail', 'middle', 'tip']`.

    position: array_like, shape (3,), default=(0,0,0)
        Reference position of the vertices in the global CS. The zero position is
        in the barycenter of the vertices.

    orientation: scipy Rotation object with length 1 or m, default=`identity`
        Orientation of the vertices in the global CS.

    Returns
    -------
    3D-model: dict
        A dictionary with necessary key/value pairs with the necessary information to construct
        a 3D-model.
    """
    pivot_conditions = {
        "tail": height / 2,
        "tip": -height / 2,
        "middle": 0,
    }
    z_shift = validate_pivot(pivot, pivot_conditions)
    N = base
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    c = np.array([np.cos(t), np.sin(t), t * 0 - 1]) * 0.5
    tp = np.array([[0, 0, 0.5]]).T
    c = np.concatenate([c, tp], axis=1)
    c = c.T * np.array([diameter, diameter, height]) + np.array([0, 0, z_shift])
    x, y, z = c.T

    i = np.arange(N, dtype=int)
    j = i + 1
    j[-1] = 0
    k = np.array([N] * N, dtype=int)
    trace = dict(x=x, y=y, z=z, i=i, j=j, k=k)
    trace = place_and_orient_model3d(trace, orientation=orientation, position=position)
    return get_model(trace, backend=backend, show=show, scale=scale, kwargs=kwargs)


def make_Arrow(
    backend,
    base=3,
    diameter=0.3,
    height=1,
    pivot="middle",
    position=None,
    orientation=None,
    show=True,
    scale=1,
    **kwargs,
) -> dict:
    """Provides the 3D-model parameters for an arrow in dictionary form, based on
    number of vertices of the base, diameter and height. The zero position is in the
    barycenter of the vertices.

    Parameters
    ----------
    base : int, default=30
        Number of vertices of the arrow base.

    diameter : float, default=0.3
        Diameter of the arrow base.

    height : int, default=1
        Arrow height.

    pivot : str, default='middle'
        The part of the arrow that is anchored to the grid and about which it rotates.
        Can be one of `['tail', 'middle', 'tip']`.

    position: array_like, shape (3,), default=(0,0,0)
        Reference position of the vertices in the global CS. The zero position is
        in the barycenter of the vertices.

    orientation: scipy Rotation object with length 1 or m, default=`identity`
        Orientation of the vertices in the global CS.

    Returns
    -------
    3D-model: dict
        A dictionary with necessary key/value pairs with the necessary information to construct
        a 3D-model.
    """

    h, d, z = height, diameter, 0
    pivot_conditions = {
        "tail": h / 2,
        "tip": -h / 2,
        "middle": 0,
    }
    z = validate_pivot(pivot, pivot_conditions)
    cone = make_Pyramid(
        backend="plotly-dict",
        base=base,
        diameter=d,
        height=d,
        position=(0, 0, z + h / 2 - d / 2),
    )
    prism = make_Prism(
        backend="plotly-dict",
        base=base,
        diameter=d / 2,
        height=h - d,
        position=(0, 0, z + -d / 2),
    )
    trace = merge_mesh3d(cone, prism)
    trace = place_and_orient_model3d(trace, orientation=orientation, position=position)
    return get_model(trace, backend=backend, show=show, scale=scale, kwargs=kwargs)
