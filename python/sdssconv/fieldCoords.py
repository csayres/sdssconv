import numpy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

SMALL_NUM = 1e-11 # epsilon for computational accuracy


def fieldAngle2Cart(fieldXY):
    """Convert (ZEMAX-style) field angles in degrees to a cartesian point on
    the unit sphere of the sky

    +X is aligned with +RA
    +Y is aligned with +DEC
    +Z is aligned with optical axis and points out of telescope toward sky

    Zemax defines field angles like this:
    Positive field angles imply positive slope for the ray in that direction,
    and thus refer to **negative** coordinates on distant objects. ZEMAX converts
    x field angles ( αx ) and y field angles ( αy ) to ray direction cosines
    using the following formulas:

    tanαx = l/n
    tanαy = m/n
    l^2 + m^2 + n^2 = 1

    where l, m, and n are the x, y, and z direction cosines.

    Parameters
    -----------
    fieldXY: 2D array
        [x, y] field angles in degrees

    Returns
    ---------
    result: numpy.array
        x,y,z coordinates on unit sphere
    """
    xField, yField = fieldXY

    # invert field degrees, fields represent slope
    # of ray, so a positivly sloped ray is coming from
    # below its respective plane, the resulting vector
    # from this transform should thus point downward, not up
    tanx = numpy.tan(numpy.radians(-1*xField))
    tany = numpy.tan(numpy.radians(-1*yField))

    # calculate direction cosines
    n = numpy.sqrt(1/(tanx**2 + tany**2 + 1))
    l = tanx * n
    m = tany * n
    if numpy.isnan(n) or numpy.isnan(l) or numpy.isnan(m):
        raise RuntimeError("NaN output [%.2f, %.2f, %.2f] for input [%.2f, %.2f]"%(n, l, m, xField, yField))

    return numpy.array([l, m, n])


def cart2FieldAngle(cartXYZ):
    """Convert cartesian point on unit sphere of sky to (ZEMAX-style) field
    angles in degrees.

    +X is aligned with +RA
    +Y is aligned with +DEC
    +Z is aligned with optical axis and points out of telescope toward sky

    Zemax defines field angles like this:
    Positive field angles imply positive slope for the ray in that direction,
    and thus refer to **negative** coordinates on distant objects. ZEMAX converts
    x field angles ( αx ) and y field angles ( αy ) to ray direction cosines
    using the following formulas:

    tanαx = l/n
    tanαy = m/n
    l^2 + m^2 + n^2 = 1

    where l, m, and n are the x, y, and z direction cosines.

    Parameters
    -----------
    cartXYZ: numpy.array
        x,y,z coordinates on unit sphere

    Returns
    ---------
    result: numpy.array
        [xField, yField] ZEMAX style field coordinates (degrees)
    """
    if numpy.abs(numpy.linalg.norm(cartXYZ) - 1) > SMALL_NUM:
        raise RuntimeError("cartXYZ must be a vector on unit sphere with L2 norm = 1")
    l, m, n = cartXYZ

    # invert field degrees, fields represent slope
    # of ray, so a positivly sloped ray is coming from
    # below the optical axis, the resulting vector
    # from this transform should thus point downward, not up

    xField = -1*numpy.degrees(numpy.arctan2(l, n))
    yField = -1*numpy.degrees(numpy.arctan2(m, n))
    if numpy.isnan(xField) or numpy.isnan(yField):
        raise RuntimeError("NaN output [%.2f, %.2f] for input [%.2f, %.2f, %.2f]"%(xField, yField, cartXYZ[0], cartXYZ[1], cartXYZ[2]))
    return numpy.array([xField, yField])


def cart2Sph(cartXYZ):
    """Convert cartesian coordinates on unit sphere to spherical
    coordinates theta, phi in degrees

    Input represents a direction on the unit sphere with
    +X is aligned with +RA
    +Y is aligned with +DEC
    +Z is aligned with optical axis and points out of telescope toward sky

    theta is the positive angle from +X toward +Y
    phi is the angle from +Z toward XY plane, phi can be thought as the degree off axis
    of a target

    Parameters
    -----------
    cartXYZ: numpy.array
        x,y,z coordinates on unit sphere

    Returns
    ---------
    result: numpy.array
        [phi, theta] degrees
    """

    if numpy.abs(numpy.linalg.norm(cartXYZ) - 1) > SMALL_NUM:
        raise RuntimeError("cartXYZ must be a vector on unit sphere with L2 norm = 1")
    if cartXYZ[2] < 0:
        raise RuntimeError("z direction must be positive for cartesian field coord")
    x, y, z = cartXYZ
    theta = numpy.degrees(numpy.arctan2(y, x))
    # wrap theta to be between 0 and 360 degrees
    if theta < 0:
        theta += 360
    phi = numpy.degrees(numpy.arccos(z))
    if numpy.isnan(theta) or numpy.isnan(phi):
        raise RuntimeError("NaN output [%.2f, %.2f] from input [%.2f, %.2f, %.2f]"%(theta, phi, cartXYZ[0], cartXYZ[1], cartXYZ[2]))
    return numpy.array([phi, theta])


def sph2Cart(phiTheta):
    """Convert spherical coordinates theta, phi in degrees
    to cartesian coordinates on unit sphere.

    Input represents a direction on the unit sphere with
    +X is aligned with +RA
    +Y is aligned with +DEC
    +Z is aligned with optical axis and points out of telescope toward sky

    theta is the positive angle from +X toward +Y
    phi is the angle from +Z toward XY plane, phi can be though as the degree off axis
    of a target

    Parameters
    -----------
    phiTheta: array
        [phi, theta] coordinates in degrees

    Returns
    ---------
    result: numpy.array
        [x,y,z] coordinates on unit sphere

    """
    phi, theta = phiTheta
    if theta < 0 or theta >= 360:
        raise RuntimeError("theta must be in range [0, 360]")
    if phi < 0 or phi >= 90:
        raise RuntimeError("phi must be in range [0, 90]")
    theta, phi = numpy.radians(theta), numpy.radians(phi)
    x = numpy.cos(theta) * numpy.sin(phi)
    y = numpy.sin(theta) * numpy.sin(phi)
    z = numpy.cos(phi)
    if numpy.isnan(x) or numpy.isnan(y) or numpy.isnan(z):
        raise RuntimeError("NaN output [%.2f, %.2f, %.2f] for input [%.2f, %.2f]"%(x, y, z, numpy.degrees(theta), numpy.degrees(phi)))
    return numpy.array([x, y, z])

