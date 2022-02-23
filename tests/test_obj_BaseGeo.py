import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.exceptions import MagpylibBadUserInput
import magpylib as magpy


def test_BaseGeo_basics():
    """fundamental usage test"""
    # pylint: disable=no-member

    ptest = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0.67545246, -0.6675014, -0.21692852],
        ]
    )

    otest = np.array(
        [
            [0, 0, 0],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0, 0, 0],
            [0.20990649, 0.41981298, 0.62971947],
            [0, 0, 0],
            [0.59199676, 0.44281248, 0.48074693],
        ]
    )

    poss, rots = [], []

    bgeo = BaseGeo((0, 0, 0), None)
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    bgeo.position = (1, 2, 3)
    bgeo.orientation = R.from_rotvec((0.1, 0.2, 0.3))
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    bgeo.move((-1, -2, -3))
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    rot = R.from_rotvec((-0.1, -0.2, -0.3))
    bgeo.rotate(rotation=rot, start=-1)
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    bgeo.rotate_from_angax(angle=45, axis=(1, 2, 3))
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    bgeo.rotate_from_angax(-np.pi / 4, (1, 2, 3), degrees=False)
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    rot = R.from_rotvec((0.1, 0.2, 0.3))
    bgeo.rotate(rot, anchor=(3, 2, 1), start=-1)
    bgeo.rotate_from_angax(33, (3, 2, 1), anchor=0, start=-1)
    poss += [bgeo.position.copy()]
    rots += [bgeo.orientation.as_rotvec()]

    poss = np.array(poss)
    rots = np.array(rots)

    assert np.allclose(poss, ptest), "test_BaseGeo bad position"
    assert np.allclose(rots, otest), "test_BaseGeo bad orientation"


def test_rotate_vs_rotate_from():
    """testing rotate vs rotate_from_angax"""
    roz = [
        (0.1, 0.2, 0.3),
        (-0.1, -0.1, -0.1),
        (0.2, 0, 0),
        (0.3, 0, 0),
        (0, 0, 0.4),
        (0, -0.2, 0),
    ]

    bg1 = BaseGeo(position=(3, 4, 5), orientation=R.from_quat((0, 0, 0, 1)))
    for ro in roz:
        rroz = R.from_rotvec((ro,))
        bg1.rotate(rotation=rroz, anchor=(-3, -2, 1))
    pos1 = bg1.position
    ori1 = bg1.orientation.as_quat()

    bg2 = BaseGeo(position=(3, 4, 5), orientation=R.from_quat((0, 0, 0, 1)))
    angs = np.linalg.norm(roz, axis=1)
    for ang, ax in zip(angs, roz):
        bg2.rotate_from_angax(angle=[ang], degrees=False, axis=ax, anchor=(-3, -2, 1))
    pos2 = bg2.position
    ori2 = bg2.orientation.as_quat()

    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)


def test_BaseGeo_reset_path():
    """testing reset path"""
    # pylint: disable=protected-access
    bg = BaseGeo((0, 0, 0), R.from_quat((0, 0, 0, 1)))
    bg.move([(1, 1, 1)] * 11)

    assert len(bg._position) == 12, "bad path generation"

    bg.reset_path()
    assert len(bg._position) == 1, "bad path reset"


def test_BaseGeo_anchor_None():
    """testing rotation with None anchor"""
    pos = np.array([1, 2, 3])
    bg = BaseGeo(pos, R.from_quat((0, 0, 0, 1)))
    bg.rotate(R.from_rotvec([(0.1, 0.2, 0.3), (0.2, 0.4, 0.6)]))

    pos3 = np.array([pos] * 3)
    rot3 = np.array([(0, 0, 0), (0.1, 0.2, 0.3), (0.2, 0.4, 0.6)])
    assert np.allclose(bg.position, pos3), "None rotation changed position"
    assert np.allclose(
        bg.orientation.as_rotvec(), rot3
    ), "None rotation did not adjust rot"


def evall(obj):
    """return pos and orient of object"""
    # pylint: disable=protected-access
    pp = obj._position
    rr = obj._orientation.as_quat()
    rr = np.array([r / max(r) for r in rr])
    return (pp, rr)


def test_attach():
    """test attach functionality"""
    bg = BaseGeo([0, 0, 0], R.from_rotvec((0, 0, 0)))
    rot_obj = R.from_rotvec([(x, 0, 0) for x in np.linspace(0, 10, 11)])
    bg.rotate(rot_obj, start=-1)

    bg2 = BaseGeo([0, 0, 0], R.from_rotvec((0, 0, 0)))
    roto = R.from_rotvec(((1, 0, 0),))
    for _ in range(10):
        bg2.rotate(roto)

    assert np.allclose(bg.position, bg2.position), "attach p"
    assert np.allclose(bg.orientation.as_quat(), bg2.orientation.as_quat()), "attach o"


def test_path_functionality1():
    """testing path functionality in detail"""
    pos0 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5.0]])
    rot0 = R.from_quat(
        [(1, 0, 0, 1), (2, 0, 0, 1), (4, 0, 0, 1), (5, 0, 0, 1), (10, 0, 0, 1.0)]
    )
    inpath = np.array([(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3)])

    b1, b2, b3, b4, b5 = pos0
    c1, c2, c3 = inpath
    q1, q2, q3, q4, q5 = np.array(
        [(1, 0, 0, 1), (1, 0, 0, 0.5), (1, 0, 0, 0.25), (1, 0, 0, 0.2), (1, 0, 0, 0.1)]
    )

    pos, ori = evall(BaseGeo(pos0, rot0))
    P = np.array([b1, b2, b3, b4, b5])
    Q = np.array([q1, q2, q3, q4, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=0))
    P = np.array([b1 + c1, b2 + c2, b3 + c3, b4, b5])
    Q = np.array([q1, q2, q3, q4, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=1))
    P = np.array([b1, b2 + c1, b3 + c2, b4 + c3, b5])
    Q = np.array([q1, q2, q3, q4, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=2))
    P = np.array([b1, b2, b3 + c1, b4 + c2, b5 + c3])
    Q = np.array([q1, q2, q3, q4, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)


def test_path_functionality2():
    """testing path functionality in detail"""
    pos0 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5.0]])
    rot0 = R.from_quat(
        [(1, 0, 0, 1), (2, 0, 0, 1), (4, 0, 0, 1), (5, 0, 0, 1), (10, 0, 0, 1.0)]
    )
    inpath = np.array([(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3)])

    b1, b2, b3, b4, b5 = pos0
    c1, c2, c3 = inpath
    q1, q2, q3, q4, q5 = np.array(
        [(1, 0, 0, 1), (1, 0, 0, 0.5), (1, 0, 0, 0.25), (1, 0, 0, 0.2), (1, 0, 0, 0.1)]
    )

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=3))
    P = np.array([b1, b2, b3, b4 + c1, b5 + c2, b5 + c3])
    Q = np.array([q1, q2, q3, q4, q5, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=4))
    P = np.array([b1, b2, b3, b4, b5 + c1, b5 + c2, b5 + c3])
    Q = np.array([q1, q2, q3, q4, q5, q5, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=5))
    P = np.array([b1, b2, b3, b4, b5, b5 + c1, b5 + c2, b5 + c3])
    Q = np.array([q1, q2, q3, q4, q5, q5, q5, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath, start=5))
    P = np.array([b1, b2, b3, b4, b5, b5 + c1, b5 + c2, b5 + c3])
    Q = np.array([q1, q2, q3, q4, q5, q5, q5, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)

    pos, ori = evall(BaseGeo(pos0, rot0).move(inpath))
    P = np.array([b1, b2, b3, b4, b5, b5 + c1, b5 + c2, b5 + c3])
    Q = np.array([q1, q2, q3, q4, q5, q5, q5, q5])
    assert np.allclose(pos, P)
    assert np.allclose(ori, Q)


def test_path_functionality3():
    """testing path functionality in detail"""
    pos0 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5.0]])
    rot0 = R.from_quat(
        [(1, 0, 0, 1), (2, 0, 0, 1), (4, 0, 0, 1), (5, 0, 0, 1), (10, 0, 0, 1.0)]
    )
    inpath = np.array([(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3)])

    pos1, ori1 = evall(BaseGeo(pos0, rot0).move(inpath, start=4))
    pos2, ori2 = evall(BaseGeo(pos0, rot0).move(inpath, start=-1))
    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)

    pos1, ori1 = evall(BaseGeo(pos0, rot0).move(inpath, start=3))
    pos2, ori2 = evall(BaseGeo(pos0, rot0).move(inpath, start=-2))
    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)

    pos1, ori1 = evall(BaseGeo(pos0, rot0).move(inpath, start=2))
    pos2, ori2 = evall(BaseGeo(pos0, rot0).move(inpath, start=-3))
    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)

    pos1, ori1 = evall(BaseGeo(pos0, rot0).move(inpath, start=1))
    pos2, ori2 = evall(BaseGeo(pos0, rot0).move(inpath, start=-4))
    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)

    pos1, ori1 = evall(BaseGeo(pos0, rot0).move(inpath, start=0))
    pos2, ori2 = evall(BaseGeo(pos0, rot0).move(inpath, start=-5))
    assert np.allclose(pos1, pos2)
    assert np.allclose(ori1, ori2)


def test_scipy_from_methods():
    """test all rotation methods inspired from scipy implemented in BaseTransform"""
    cube = lambda: magpy.magnet.Cuboid((11, 22, 33), (1, 1, 1))
    angs_deg = np.linspace(0, 360, 10)
    angs = np.deg2rad(angs_deg)
    rot = R.from_rotvec((np.array([[0, 0, 1]] * 10).T * angs).T)
    anchor = (1, 2, 3)
    cube0 = cube().rotate(rot, anchor=anchor)

    from_rotvec = cube().rotate_from_rotvec(
        rot.as_rotvec(degrees=True), anchor=anchor, degrees=True
    )
    assert np.allclose(
        cube0.position, from_rotvec.position
    ), "from_rotvec failed on position"
    assert np.allclose(
        cube0.orientation.as_rotvec(), from_rotvec.orientation.as_rotvec()
    ), "from_rotvec failed on orientation"

    from_angax = cube().rotate_from_angax(angs_deg, "z", anchor=anchor, degrees=True)
    assert np.allclose(
        cube0.position, from_angax.position
    ), "from_angax failed on position"
    assert np.allclose(
        cube0.orientation.as_rotvec(), from_angax.orientation.as_rotvec()
    ), "from_rotvec failed on orientation"

    from_euler = cube().rotate_from_euler(angs_deg, "z", anchor=anchor, degrees=True)
    assert np.allclose(
        cube0.position, from_euler.position
    ), "from_euler failed on position"
    assert np.allclose(
        cube0.orientation.as_rotvec(), from_euler.orientation.as_rotvec()
    ), "from_rotvec failed on orientation"

    from_matrix = cube().rotate_from_matrix(rot.as_matrix(), anchor=anchor)
    assert np.allclose(
        cube0.position, from_matrix.position
    ), "from_matrix failed on position"
    assert np.allclose(
        cube0.orientation.as_rotvec(), from_matrix.orientation.as_rotvec()
    ), "from_rotvec failed on orientation"

    from_mrp = cube().rotate_from_mrp(rot.as_mrp(), anchor=anchor)
    assert np.allclose(cube0.position, from_mrp.position), "from_mrp failed on position"
    assert np.allclose(
        cube0.orientation.as_rotvec(), from_mrp.orientation.as_rotvec()
    ), "from_rotvec failed on orientation"

    from_quat = cube().rotate_from_quat(rot.as_quat(), anchor=anchor)
    assert np.allclose(
        cube0.position, from_quat.position
    ), "from_quat failed on position"
    assert np.allclose(
        cube0.orientation.as_rotvec(), from_quat.orientation.as_rotvec()
    ), "from_rotvec failed on orientation"


def test_style():
    """test when setting wrong style class"""
    bg = BaseGeo((0, 0, 0), None)
    with pytest.raises(ValueError):
        bg.style = "wrong class"


def test_kwargs():
    """test kwargs inputs, only relevant for styles"""
    bg = BaseGeo((0, 0, 0), None, style=dict(label="label_01"), style_label="label_02")
    assert bg.style.label == "label_02"

    with pytest.raises(TypeError):
        bg = BaseGeo((0, 0, 0), None, styl_label="label_02")


def test_bad_sum():
    """test when adding bad objects"""
    cuboid = magpy.magnet.Cuboid((1, 1, 1), (1, 1, 1))
    with pytest.raises(MagpylibBadUserInput):
        1 + cuboid


def test_copy():
    """test copying object"""
    bg1 = BaseGeo((0, 0, 0), None, style_label='label1') #has style
    bg2 = BaseGeo((1,2,3), None) # has no style
    bg3 = BaseGeo((4,6,8), style_color='blue') # has style but label is None
    bg1c = bg1.copy()
    bg2c = bg2.copy(position=(10, 0, 0), style=dict(color='red'), style_color='orange')
    bg3c = bg3.copy()

    # original object should not be affected"
    np.testing.assert_allclose(bg1.position, (0, 0, 0))
    np.testing.assert_allclose(bg2.position, (1 ,2, 3))

    # check if label suffix iterated correctly
    assert bg1c.style.label == "label2"
    assert bg2c.style.label is None
    assert bg3c.style.label == "BaseGeo_01"

    # check if style is passed correctly
    assert bg2c.style.color == "orange"
