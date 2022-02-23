import magpylib as magpy

def test_bare_init():
    """test if magpylib object can be initilized without attributes"""
    magpy.current.Loop()
    magpy.current.Line()
    magpy.magnet.Cuboid()
    magpy.magnet.Cylinder()
    magpy.magnet.CylinderSegment()
    magpy.magnet.Sphere()
    magpy.misc.Dipole()
    magpy.misc.CustomSource()
