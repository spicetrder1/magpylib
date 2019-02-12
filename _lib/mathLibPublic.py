# -*- coding: utf-8 -*-

import numpy as np
from numpy import cos, sin, array, arccos, float64, pi
from magPyLib._lib.mathLibPrivate import getPhi, fastNorm3D, angleAxisRotation



def randomAxis():
    """
    This function generates a random `axis` (3-vector of length 1) from an equal
    angular distribution using a MonteCarlo scheme.
       
    Parameters:
    ----------
    none
        
    Returns:    
    --------
    axis : arr3
        A random axis from an equal angular distribution of length 1
        
    Example:
    --------
    >>> import magPyLib as magPy
    >>> ax = magPy.math.randomAxis()
    >>> print(ax)
      [-0.24834468  0.96858637  0.01285925]
    
    """
    while True: 
        r = np.random.rand(3)*2-1       #create random axis
        Lr2 = sum(r**2)                 #get length
        if Lr2 <= 1:                    #is axis within sphere?
            Lr = np.sqrt(Lr2)           #normalize
            return r/Lr



def axisFromAngles(angles):
    """
    This function generates an `axis` (3-vector of length 1) from two `angles` = [phi,th]
    that are defined as in spherical coordinates. phi = azimuth angle, th = polar angle.
    Vector input format can be either list, tuple or array of any data type (float, int).
       
    Parameters:
    ----------
    angles : vec2 [deg]
        The two angels [phi,th], azimuth and polar, in units of deg.
        
    Returns:    
    --------
    axis : arr3
        An axis of length that is oriented as given by the input angles.
        
    Example:
    --------
    >>> import magPyLib as magPy
    >>> angles = [90,90]
    >>> ax = magPy.math.axisFromAngles(angles)
    >>> print(ax)
      [0.0  1.0  0.0]
    """
    phi, th = angles #phi in [0,2pi], th in [0,pi]
    phi = phi/180*pi
    th = th/180*pi
    return array([cos(phi)*sin(th),sin(phi)*sin(th),cos(th)])



def anglesFromAxis(axis):
    """
    This function takes an arbitrary `axis` (3-vector) and returns the orientation
    given by the `angles` = [phi,th] that are defined as in spherical coordinates. 
    phi = azimuth angle, th = polar angle. Vector input format can be either 
    list, tuple or array of any data type (float, int).
       
    Parameters:
    ----------
    axis : vec3
        Arbitrary input axis that defines an orientation.
        
    Returns:    
    --------
    angles : arr2 [deg]
        The angles [phi,th], azimuth and polar, that correspond to the orientation 
        given by the input axis.
        
    Example:
    --------
    >>> import magPyLib as magPy
    >>> axis = [1,1,0]
    >>> angles = magPy.math.anglesFromAxis(axis)
    >>> print(angles)
      [45. 90.]
    """
    ax = array(axis, dtype=float64, copy=False)
    
    Lax = fastNorm3D(ax)
    Uax = ax/Lax
    
    TH = arccos(Uax[2])/pi*180
    PHI = getPhi(Uax[0],Uax[1])/pi*180
    return array([PHI,TH])



def rotatePosition(position,angle,axis,CoR=[0,0,0]):
    """
    This function uses angle-axis rotation to rotate the `position` vector by
    the `angle` argument about an axis defined by the `axis` vector which passes
    through the center of rotation `CoR` vector. Scalar input is either integer
    or float.Vector input format can be either list, tuple or array of any data
    type (float, int).
    
    Parameters:
    ----------
    position : vec3
        Input position to be rotated.
        
    angle : scalar [deg]
        Angle of rotation in untis of [deg]
    
    axis : vec3
        Axis of rotation

    CoR : vec3
        The Center of rotation which defines the position of the axis of rotation

    Returns:    
    --------
    newPosition : arr3
        Rotated position
        
    Example:
    --------
    >>> import magPyLib as magPy
    >>> from numpy import pi
    >>> position0 = [1,1,0]
    >>> angle = -90
    >>> axis = [0,0,1]
    >>> centerOfRotation = [1,0,0]
    >>> positionNew = magPy.math.rotatePosition(position0,angle,axis,CoR=centerOfRotation)
    >>> print(positionNew)
      [2. 0. 0.]
    """

    pos = array(position, dtype=float64, copy=False)
    ang = float(angle)
    ax = array(axis, dtype=float64, copy=False)
    cor = array(CoR, dtype=float64, copy=False)
    
    pos12 = pos-cor  
    pos12Rot = angleAxisRotation(ang,ax,pos12)
    posRot = pos12Rot+cor
    
    return posRot



