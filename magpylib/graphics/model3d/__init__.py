"""
The `magpylib.display.plotly` sub-package provides useful functions for
convenient creation of 3D traces for commonly used objects in the
library.
"""

__all__ = [
    "make_Arrow",
    "make_Ellipsoid",
    "make_Pyramid",
    "make_Cuboid",
    "make_CylinderSegment",
    "make_Prism",
]

from magpylib._src.display.base_traces import (
    make_Arrow,
    make_Ellipsoid,
    make_Pyramid,
    make_Cuboid,
    make_CylinderSegment,
    make_Prism,
)
