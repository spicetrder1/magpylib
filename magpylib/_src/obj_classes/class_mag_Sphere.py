"""Magnet Sphere class code
DOCSTRINGS V4 READY
"""

from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._src.obj_classes.class_BaseExcitations import BaseHomMag
from magpylib._src.input_checks import check_format_input_scalar

class Sphere(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseHomMag):
    """Spherical magnet with homogeneous magnetization.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the sphere center is located
    in the origin of the global coordinate system.

    Parameters
    ----------
    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector (mu0*M, remanence field) in units of [mT] given in
        the local object coordinates (rotates with object).

    diameter: float, default=`None`
        Diameter of the sphere in units of [mm].

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style-underscore_magic, e.g. `style_color='red'`.


    Returns
    -------
    magnet source: `Sphere` object

    Examples
    --------
    `Sphere` objects are magnetic field sources. In this example we compute the H-field [kA/m]
    of a spherical magnet with magnetization (100,200,300) in units of [mT] and diameter
    of 1 [mm] at the observer position (1,1,1) given in units of [mm]:

    >>> import magpylib as magpy
    >>> src = magpy.magnet.Sphere(magnetization=(100,200,300), diameter=1)
    >>> H = src.getH((1,1,1))
    >>> print(H)
    [3.19056074 2.55244859 1.91433644]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'x')
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[2.26804606 3.63693295 0.23486386]
     [0.28350576 0.45461662 0.02935798]
     [0.08400171 0.13470122 0.00869866]]

    The same result is obtained when the rotated source moves along a path away from an
    observer at position (1,1,1). This time we use a `Sensor` object as observer.

    >>> src.move([(-1,-1,-1), (-2,-2,-2)])
    >>> sens = magpy.Sensor(position=(1,1,1))
    >>> B = src.getB(sens)
    >>> print(B)
    [[2.26804606 3.63693295 0.23486386]
     [0.28350576 0.45461662 0.02935798]
     [0.08400171 0.13470122 0.00869866]]
    """

    def __init__(
        self,
        magnetization=None,
        diameter=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):

        # instance attributes
        self.diameter = diameter
        self._object_type = "Sphere"

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)
        BaseHomMag.__init__(self, magnetization)

    # property getters and setters
    @property
    def diameter(self):
        """Object diameter attribute getter and setter."""
        return self._diameter

    @diameter.setter
    def diameter(self, dia):
        """Set Sphere diameter, float, [mm]."""
        self._diameter = check_format_input_scalar(
            dia,
            sig_name='diameter',
            sig_type='`None` or a positive number (int, float)',
            allow_None=True,
            forbid_negative=True,)
