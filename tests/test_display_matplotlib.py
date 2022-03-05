import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import magpylib as magpy
from magpylib.magnet import Cylinder, Cuboid, Sphere, CylinderSegment

# pylint: disable=assignment-from-no-return

magpy.defaults.reset()

def test_Cylinder_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = Cylinder((1, 2, 3), (1, 2))
    x = src.show(canvas=ax, style_path_frames=15, backend='matplotlib')
    assert x is None, "path should revert to True"
    src.move(np.linspace((.4,.4,.4), (12,12,12), 30), start=-1)
    x = src.show(canvas=ax, style_path_show=False, show_direction=True, backend='matplotlib')
    assert x is None, "display test fail"

    x = src.show(canvas=ax, style_path_frames=[], show_direction=True, backend='matplotlib')
    assert x is None, "ind>path_len, should display last position"

    x = src.show(canvas=ax, style_path_frames=[1, 5, 6], show_direction=True, backend='matplotlib')
    assert x is None, "should display 1,5,6 position"


def test_CylinderSegment_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = CylinderSegment((1, 2, 3), (2, 4, 5, 30, 40))
    x = src.show(canvas=ax, style_path_frames=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((.4,.4,.4), (13.2,13.2,13.2), 33), start=-1)
    x = src.show(canvas=ax, style_path_show=False, show_direction=True)
    assert x is None, "display test fail"


def test_Sphere_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = Sphere((1, 2, 3), 2)
    x = src.show(canvas=ax, style_path_frames=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((.4,.4,.4), (8,8,8), 20), start=-1)
    x = src.show(canvas=ax, style_path_show=False, show_direction=True)
    assert x is None, "display test fail"


def test_Cuboid_display():
    """testing display"""
    src = Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((.1,.1,.1), (2,2,2), 20), start=-1)
    plt.ion()
    x = src.show(style_path_frames=5, show_direction=True)
    plt.close()
    assert x is None, "display test fail"

    ax = plt.subplot(projection="3d")
    x = src.show(canvas=ax, style_path_show=False, show_direction=True)
    assert x is None, "display test fail"


def test_Sensor_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    sens = magpy.Sensor(pixel=[(1, 2, 3), (2, 3, 4)])
    sens.style.arrows.z.color = 'magenta'
    sens.style.arrows.z.show = False
    poz = np.linspace((.4,.4,.4), (13.2,13.2,13.2), 33)
    sens.move(poz, start=-1)
    x = sens.show(canvas=ax, markers=[(100, 100, 100)], style_path_frames=15)
    assert x is None, "display test fail"

    x = sens.show(canvas=ax, markers=[(100, 100, 100)], style_path_show=False)
    assert x is None, "display test fail"


def test_CustomSource_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    cs = magpy.misc.CustomSource()
    x = cs.show(canvas=ax)
    assert x is None, "display test fail"


def test_Loop_display():
    """testing display for Loop source"""
    ax = plt.subplot(projection="3d")
    src = magpy.current.Loop(current=1, diameter=1)
    x = src.show(canvas=ax)
    assert x is None, "display test fail"

    src.rotate_from_angax([5] * 35, "x", anchor=(1, 2, 3))
    x = src.show(canvas=ax, style_path_frames=3)
    assert x is None, "display test fail"


def test_col_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax = plt.subplot(projection="3d")
    pm1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm2 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    col = magpy.Collection(pm1, pm2)
    x = col.show(canvas=ax)
    assert x is None, "colletion display test fail"


def test_dipole_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax2 = plt.subplot(projection="3d")
    dip = magpy.misc.Dipole(moment=(1, 2, 3), position=(2, 2, 2))
    dip2 = magpy.misc.Dipole(moment=(1, 2, 3), position=(2, 2, 2))
    dip2.move(np.linspace((.4,.4,.4), (2,2,2), 5), start=-1)
    x = dip.show(canvas=ax2)
    assert x is None, "display test fail"
    x = dip.show(canvas=ax2, style_path_frames=2)
    assert x is None, "display test fail"


def test_circular_line_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax2 = plt.subplot(projection="3d")
    src1 = magpy.current.Loop(1, 2)
    src2 = magpy.current.Loop(1, 2)
    src1.move(np.linspace((.4,.4,.4), (2,2,2), 5), start=-1)
    src3 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src4 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src3.move([(.4,.4,.4)]*5, start=-1)
    x = src1.show(canvas=ax2, style_path_frames=2, style_arrow_size=0)
    assert x is None, "display test fail"
    x = src2.show(canvas=ax2)
    assert x is None, "display test fail"
    x = src3.show(canvas=ax2, style_arrow_size=0)
    assert x is None, "display test fail"
    x = src4.show(canvas=ax2, style_path_frames=2)
    assert x is None, "display test fail"


def test_matplotlib_animation_warning():
    """animation=True with matplotlib should raise UserWarning"""
    ax = plt.subplot(projection="3d")
    sens = magpy.Sensor(pixel=[(1, 2, 3), (2, 3, 4)])
    sens.move(np.linspace((.4,.4,.4), (12.4,12.4,12.4), 33), start=-1)
    with pytest.warns(UserWarning):
        sens.show(canvas=ax, animation=True)


def test_matplotlib_model3d_extra():
    """test display extra model3d"""

    # using "plot"
    xs,ys,zs = [(1,2)]*3
    trace1 = dict(
        type='plot',
        args=(xs,ys,zs),
        ls='-',
    )
    obj1 = magpy.misc.Dipole(moment=(0,0,1))
    obj1.style.model3d.add_trace(trace=trace1, backend="matplotlib")

    # using "plot_surface"
    u, v = np.mgrid[0:2 * np.pi:6j, 0:np.pi:6j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    trace2 = dict(
        type='plot_surface',
        args=(xs,ys,zs),
        cmap=plt.cm.YlGnBu_r,
    )
    obj2 = magpy.Collection()
    obj2.style.model3d.add_trace(trace=trace2, backend='matplotlib')

    # using "plot_trisurf"
    u, v = np.mgrid[0:2*np.pi:6j, -.5:.5:6j]
    u, v = u.flatten(), v.flatten()
    xs = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
    ys = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
    zs = 0.5 * v * np.sin(u / 2.0)
    tri = mtri.Triangulation(u, v)
    trace3 = dict(
        type="plot_trisurf",
        args=(xs,ys,zs),
        triangles=tri.triangles,
        cmap=plt.cm.Spectral,
    )
    obj3 = magpy.misc.CustomSource(style_model3d_showdefault=False, position=(3,0,0))
    obj3.style.model3d.add_trace(trace=trace3, backend="matplotlib")

    magpy.show(obj1, obj2, obj3)

    ax = plt.subplot(projection="3d")
    x = magpy.show(obj1, obj2, obj3, canvas=ax)
    assert x is None, "display test fail"


def test_matplotlib_model3d_extra_bad_input():
    """test display extra model3d"""

    xs,ys,zs = [(1,2)]*3
    trace = dict(
        type='plot',
        argss=(xs,ys,zs),   # bad input
        ls='-',
    )
    obj = magpy.misc.Dipole(moment=(0,0,1))
    with pytest.raises(ValueError):
        obj.style.model3d.add_trace(trace=trace, backend="matplotlib")
        ax = plt.subplot(projection="3d")
        obj.show(canvas=ax)


def test_empty_display():
    """should not fail if nothing to display"""
    ax = plt.subplot(projection="3d")
    x = magpy.show(canvas=ax, backend="matplotlib")
    assert x is None, "empty display matplotlib test fail"

def test_subplots():
    """test subplots"""
    # define sources
    src1 = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
    src2 = magpy.magnet.Cylinder(magnetization=(0, 0, 1), dimension=(1, 2))

    # manipulate first source to create a path
    src1.move(np.linspace((0, 0, 0.1), (0, 0, 8), 20))

    # manipulate second source
    src2.move(np.linspace((0.1, 0, 0.1), (5, 0, 5), 50))
    src2.rotate_from_angax(angle=np.linspace(10, 600, 50), axis="z", anchor=0, start=1)

    # setup plotly figure and subplots
    fig = plt.figure(figsize=(12, 4))
    fig.subplots(ncols=3, subplot_kw=dict(projection='3d'))

    # draw the objects
    x = magpy.show(src1, row=1, canvas=fig)
    assert x is None, "subplots display matplotlib test fail"
    x = magpy.show(src2, col=2, canvas=fig)
    assert x is None, "subplots display matplotlib test fail"
    x = magpy.show(src1, src2, col=3, canvas=fig)
    assert x is None, "subplots display matplotlib test fail"
