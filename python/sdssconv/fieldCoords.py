import numpy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

SMALL_NUM = 1e-11 # epsilon for computational accuracy


def fieldAngle2Cart(xField, yField):
    """Convert (ZEMAX-style) field angles in degrees to a cartesian point on
    the unit sphere

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
    xField: scalar or 1D array
        zemax style x field
    yField: scalar or 1D array
        zemax style y field

    Returns
    ---------
    result: list
        [x,y,z] coordinates on unit sphere
    """
    # xField, yField = fieldXY

    # invert field degrees, fields represent slope
    # of ray, so a positivly sloped ray is coming from
    # below its respective plane, the resulting vector
    # from this transform should thus point downward, not up
    tanx = numpy.tan(numpy.radians(-1*xField))
    tany = numpy.tan(numpy.radians(-1*yField))

    # calculate direction cosines
    z = numpy.sqrt(1/(tanx**2 + tany**2 + 1))
    x = tanx * z
    y = tany * z
    # if numpy.isnan(n) or numpy.isnan(l) or numpy.isnan(m):
    #     raise RuntimeError("NaN output [%.2f, %.2f, %.2f] for input [%.2f, %.2f]"%(n, l, m, xField, yField))

    return [x,y,z]


def cart2FieldAngle(x, y, z):
    """Convert cartesian point on unit sphere of sky to (ZEMAX-style) field
    angles in degrees.


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
    x: scalar or 1D array
    y: scalar or 1D array
    z: scalar or 1D array

    Returns
    ---------
    result: list
        [xField, yField] ZEMAX style field coordinates (degrees)
    """
    # if numpy.abs(numpy.linalg.norm(cartXYZ) - 1) > SMALL_NUM:
    #     raise RuntimeError("cartXYZ must be a vector on unit sphere with L2 norm = 1")
    # l, m, n = x, y, z

    # invert field degrees, fields represent slope
    # of ray, so a positivly sloped ray is coming from
    # below the optical axis, the resulting vector
    # from this transform should thus point downward, not up

    xField = -1*numpy.degrees(numpy.arctan2(x, z))
    yField = -1*numpy.degrees(numpy.arctan2(y, z))
    # if numpy.isnan(xField) or numpy.isnan(yField):
    #     raise RuntimeError("NaN output [%.2f, %.2f] for input [%.2f, %.2f, %.2f]"%(xField, yField, cartXYZ[0], cartXYZ[1], cartXYZ[2]))
    return [xField, yField]


def cart2Sph(x, y, z):
    """Convert cartesian coordinates to spherical
    coordinates theta, phi in degrees

    phi is polar angle measure from z axis
    theta is azimuthal angle measured from x axis

    Parameters
    -----------
    x: scalar or 1D array
    y: scalar or 1D array
    z: scalar or 1D array

    Returns
    ---------
    result: list
        [theta, phi] degrees
    """

    # if numpy.abs(numpy.linalg.norm(cartXYZ) - 1) > SMALL_NUM:
    #     raise RuntimeError("cartXYZ must be a vector on unit sphere with L2 norm = 1")
    # if cartXYZ[2] < 0:
    #     raise RuntimeError("z direction must be positive for cartesian field coord")
    # x, y, z = cartXYZ
    theta = numpy.degrees(numpy.arctan2(y, x))
    # wrap theta to be between 0 and 360 degrees
    try:
        if theta < 0:
            theta += 360
    except:
        # theta is array
        inds = numpy.argwhere(theta < 0)
        theta[inds] = theta[inds] + 360
    phi = numpy.degrees(numpy.arccos(z))
    # if numpy.isnan(theta) or numpy.isnan(phi):
    #     raise RuntimeError("NaN output [%.2f, %.2f] from input [%.2f, %.2f, %.2f]"%(theta, phi, cartXYZ[0], cartXYZ[1], cartXYZ[2]))
    return [theta, phi]


def sph2Cart(theta, phi, r=1):
    """Convert spherical coordinates theta, phi in degrees
    to cartesian coordinates on unit sphere.

    phi is polar angle measure from z axis
    theta is azimuthal angle measured from x axis

    Parameters
    -----------
    theta: scalar or 1D array
        degrees, azimuthal angle
    phi: scalar or 1D array
        degrees, polar angle
    r: scalar
        radius of curvature. Default to 1 for unit sphere

    Returns
    ---------
    result: list
        [x,y,z] coordinates on sphere

    """
    # theta, phi = thetaPhi
    # while theta < 0:
    #     theta += 360
    # while theta >= 360:
    #     theta -= 360
    # if theta < 0 or theta >= 360:
    #     raise RuntimeError("theta must be in range [0, 360]")

    # if phi < -90 or phi > 90:
    #     raise RuntimeError("phi must be in range [-90, 90]")

    theta, phi = numpy.radians(theta), numpy.radians(phi)
    x = r*numpy.cos(theta) * numpy.sin(phi)
    y = r*numpy.sin(theta) * numpy.sin(phi)
    z = r*numpy.cos(phi)

    # if numpy.isnan(x) or numpy.isnan(y) or numpy.isnan(z):
    #     raise RuntimeError("NaN output [%.2f, %.2f, %.2f] for input [%.2f, %.2f]"%(x, y, z, numpy.degrees(theta), numpy.degrees(phi)))

    return [x, y, z]


def azAlt2HaDec(az, alt, latitude):
    """Convert Az/Alt to Hour angle and Declination

    Parameters
    -----------
    az: scalar or 1D array
        degrees az = 0 north az=90 east
    alt: scalar or 1D array
        degrees
    latitude: float
        degrees latitude of observer, positive for north

    Returns:
    ----------
    result : list
        [hour angle, declination] degrees
    """
    az, alt, latitude = numpy.radians(az), numpy.radians(alt), numpy.radians(latitude)

    # convert to Meeus convention for AzAlt where
    # Az = 0 is south, Az = 90 is west
    az = az - numpy.pi

    sinAz = numpy.sin(az)
    cosAz = numpy.cos(az)
    sinAlt = numpy.sin(alt)
    cosAlt = numpy.cos(alt)
    tanAlt = numpy.tan(alt)
    sinLat = numpy.sin(latitude)
    cosLat = numpy.cos(latitude)

    ha = numpy.degrees(numpy.arctan2(sinAz, cosAz*sinLat + tanAlt*cosLat))
    dec = numpy.degrees(numpy.arcsin(sinLat*sinAlt - cosLat*cosAlt*cosAz))
    return [ha, dec]


def haDec2AzAlt(ha, dec, latitude):
    """Convert Az/Alt to Hour angle and Declination

    Parameters
    -----------
    ha : scalar or 1D array
        hour angle degrees
    dec: scalar or 1D array
        declination degrees
    latitude: scalar
        degrees latitude of observer, positive for north

    Returns:
    ----------
    result : list
        [az, alt] degrees Az=0 is north Az=90 is east.
    """

    ha, dec, latitude = [numpy.radians(x) for x in [ha, dec, latitude]]
    sinHa = numpy.sin(ha)
    cosHa = numpy.cos(ha)
    sinDec = numpy.sin(dec)
    cosDec = numpy.cos(dec)
    tanDec = numpy.tan(dec)
    sinLat = numpy.sin(latitude)
    cosLat = numpy.cos(latitude)

    # meeus routine, uses convention of az 0 is south
    az = numpy.degrees(numpy.arctan2(sinHa, cosHa*sinLat - tanDec*cosLat))
    # convert to az 0 is north
    az = az + 180

    # wrap between 0 and 360
    try:
        if az < 0:
            az += 360
        if az >= 360:
            az -= 360
    except:
        # az is an array
        inds = numpy.argwhere(az<0)
        az[inds] = az[inds] + 360
        inds = numpy.argwhere(az>=360)
        az[inds] = az[inds] - 360

    alt = numpy.degrees(numpy.arcsin(sinLat*sinDec + cosLat*cosDec*cosHa))
    return [az, alt]


def parallacticAngle(ha, dec, latitude):
    """Calculate the parallacticAngle q

    q is the angle between zenith and north.  Looking south an object with
    positive hour angle with have positive q, object with negative
    hour angle will have negative q.  Note q isn't defined for an object
    at the zenith!

    So looking south setting objects appear rotated clockwise by q,
    rising objects appear rotated counter clockwise by q,
    objects on the meridian have q=0 (zenith aligned with north).

    Parameters
    -----------
    ha : scalar or 1D array
        hour angle degrees
    dec: scalar or 1D array
        declination degrees
    latitude: scalar
        degrees latitude of observer, positive for north

    Returns:
    ----------
    result : scalar
        parallactic angle in degrees
    """
    ha, dec, latitude = [numpy.radians(x) for x in [ha, dec, latitude]]
    sinHa = numpy.sin(ha)
    cosHa = numpy.cos(ha)
    sinDec = numpy.sin(dec)
    cosDec = numpy.cos(dec)
    tanLat = numpy.tan(latitude)
    q = numpy.arctan2(sinHa, tanLat*cosDec - sinDec*cosHa)
    return numpy.degrees(q)


def observedToField(az, alt, azCenter, altCenter, latitude):
    """Convert Az/Alt coordinates to cartesian coords on the
    unit sphere with the z axis aligned with field center (boresight)

    Az=0 is north, Az=90 is east

    Resulting Cartesian coordinates have the following
    convention +x points along +RA, +y points along +Dec

    Parameters
    ------------
    az: scalar or 1D array
        azimuth in degrees az=0 is north az=90 is east
    alt: scalar or 1D array
        altitude in degrees, positive above horizon, negative below
    azCenter: scalar
        azimuth in degrees, field center
    altCenter: scalar
        altitude in degrees, field center
    latitude: scalar
        observer latitude in degrees, positive for north

    Returns
    --------
    result: list
        [x,y,z] coordinates on unit sphere, with boresight (+z)
        pointed at az/alt center.  +x aligned with +RA, +y
        aligned with +DEC
    """
    # convert azAlt and azAltCen to a spherical coord sys
    # az increases westward
    # where x is aligned with Az=0, due south
    # and y is aligned with Az=-90
    # and z points to zenith

    # add some error handling?

    # az increases clockwise looking down
    # but angles are measured counterclockwise
    # in sph sys
    # azAlt = numpy.array([az, alt]).T

    thetas = -1*az
    phis = 90 - alt
    # thetaCen = -1*azCenter
    # phiCen = 90 - altCenter
    coords = sph2Cart(thetas, phis)
    coords = numpy.array(coords).T

    # coords = []
    # for theta, phi in zip(thetas, phis):
    #     # really outta parallelize this
    #     coords.append(sph2Cart([theta, phi]))
    # coords = numpy.array(coords)

    # coordCen = sph2Cart([thetaCen, phiCen])

    # rotate the xyz coordinate system about z axis
    # such that -y axis is aligned with the azimuthal angle
    # of the field center

    sinTheta = numpy.sin(numpy.radians(90-azCenter))
    cosTheta = numpy.cos(numpy.radians(90-azCenter))
    rotTheta = numpy.array([
        [ cosTheta, sinTheta, 0],
        [-sinTheta, cosTheta, 0],
        [        0,        0, 1]
    ])

    coords = rotTheta.dot(coords.T).T

    # rotate the xyz coordinate system about the x axis
    # such that +z points to the field center.

    sinPhi = numpy.sin(numpy.radians(90-altCenter))
    cosPhi = numpy.cos(numpy.radians(90-altCenter))
    rotPhi = numpy.array([
        [1,       0,      0],
        [0,  cosPhi, sinPhi],
        [0, -sinPhi, cosPhi]
    ])
    coords = rotPhi.dot(coords.T).T
    # return coords

    # finally rotate about z by the parallactic angle
    # this puts +RA along +X and +DEC along +Y
    ha, dec = azAlt2HaDec(azCenter, altCenter, latitude)
    q = parallacticAngle(ha, dec, latitude)
    cosQ = numpy.cos(numpy.radians(q))
    sinQ = numpy.sin(numpy.radians(q))
    rotQ = numpy.array([
        [ cosQ, sinQ, 0],
        [-sinQ, cosQ, 0],
        [    0,    0, 1]
    ])

    coords = rotQ.dot(coords.T).T

    if len(coords.shape) == 1:
        #single coord fed in
        return coords[0], coords[1], coords[2]
    else:
        # multiple coords fed in
        # return arrays instead
        return coords[:, 0], coords[:, 1], coords[:, 2]


def fieldToObserved(x, y, z, azCenter, altCenter, latitude):
    """Convert xyz unit-spherical field coordinates to Az/Alt

    the inverse of observetToField

    Az=0 is north, Az=90 is east

    +x points along +RA, +y points along +Dec, +z points along boresight
    from telescope toward sky

    Parameters
    ------------
    x: scalar or 1D array
        unit-spherical x field coord (aligned with +RA)
    y: scalar or 1D array
        unit-spherical y field coord (aligned with +Dec)
    z: scalar or 1D array
        unit-spherical z coord
    azCenter: scalar
        azimuth in degrees, field center
    altCenter: scalar
        altitude in degrees, field center
    latitude: scalar
        observer latitude in degrees, positive for north

    Returns
    --------
    result: list
        az, alt.  Az=0 is north, Az=90 is east
    """
    # note this is basically cut and paste from
    # observedToField with rotations inverted
    ha, dec = azAlt2HaDec(azCenter, altCenter, latitude)
    q = parallacticAngle(ha, dec, latitude)
    cosQ = numpy.cos(numpy.radians(-1*q))
    sinQ = numpy.sin(numpy.radians(-1*q))
    rotQ = numpy.array([
        [ cosQ, sinQ, 0],
        [-sinQ, cosQ, 0],
        [    0,    0, 1]
    ])
    coords = numpy.array([x,y,z]).T
    coords = rotQ.dot(coords.T).T


    sinPhi = numpy.sin(-1*numpy.radians(90-altCenter))
    cosPhi = numpy.cos(-1*numpy.radians(90-altCenter))
    rotPhi = numpy.array([
        [1,       0,      0],
        [0,  cosPhi, sinPhi],
        [0, -sinPhi, cosPhi]
    ])
    coords = rotPhi.dot(coords.T).T

    sinTheta = numpy.sin(-1*numpy.radians(90-azCenter))
    cosTheta = numpy.cos(-1*numpy.radians(90-azCenter))
    rotTheta = numpy.array([
        [ cosTheta, sinTheta, 0],
        [-sinTheta, cosTheta, 0],
        [        0,        0, 1]
    ])

    coords = rotTheta.dot(coords.T).T

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    # thetaCen = -1*azCenter
    # phiCen = 90 - altCenter
    theta, phi = cart2Sph(x, y, z)

    # convert sph theta, phi to az, alt
    az = -1 * theta
    alt = 90 - phi

    if len(az) == 1:
        #single coord fed in
        az, alt = az[0], alt[0]
        if az < 0:
            az += 360
        if az >= 360:
            az -= 360
        return az, alt
    else:
        # multiple coords fed in
        # return arrays instead
        inds = numpy.argwhere(az < 0)
        az[inds] = az[inds] + 360
        inds = numpy.argwhere(az >= 360)
        az[inds] = az[inds] - 360
        return az, alt


class FocalPlaneModel(object):
    """A class that holds parameters usefull for converting between
    focal plane and field coordinates

    note: should probably include limits of valid interpolations
    """
    def __init__(self, r, b, powers, forwardCoeffs, reverseCoeffs):
        """
        Parameters
        -----------
        r : float
            radius of curvature (mm)
        b : float
            center of curvature (mm) along positive z axis
        powers: list of ints
            the powers to be included in the polynomial fit
        forwardCoeffs: list of floats
            coefficents associated with power list for field to
            focal distortion model
        reverseCoeffs: list of floats
            coefficients associated with power list for focal to
            field distortion model
        """
        self.r = r
        self.b = b
        self.powers = powers
        self.forwardCoeffs = forwardCoeffs
        self.reverseCoeffs = reverseCoeffs

    def _forwardDistort(self, phiField):
        """Convert off axis angle on field to off axis angle on focal plane
        """
        if hasattr(phiField, "__len__"):
            phiField = numpy.array(phiField)
            phiFocal = numpy.zeros(len(phiField))
        else:
            phiFocal = 0

        for p, c in zip(self.powers, self.forwardCoeffs):
            phiFocal += c * phiField ** p

        return phiFocal

    def _reverseDistort(self, phiFocal):
        """Convert off axis angle on focal plane to off axis angle on field
        """
        if hasattr(phiFocal, "__len__"):
            phiFocal = numpy.array(phiFocal)
            phiField = numpy.zeros(len(phiFocal))
        else:
            phiField = 0

        for p, c in zip(self.powers, self.reverseCoeffs):
            phiField += c * phiFocal ** p

        return phiField

    def fieldToFocal(self, xField, yField, zField):
        """Convert unit-spherical xyz field to non-unit spherical position on
        the focal plane.  Focal plane origin is the M1 vertex.

        Parameters
        -----------
        xField: scalar or 1D array
            unit-spherical x field coord (aligned with +RA on sky)
        yField: scalar or 1D array
            unit-spherical y field coord (aligned with +Dec on sky)
        zField: scalar or 1D array
            unit-spherical z coord

        Result
        -------
        xFocal: scalar or 1D array
            spherical x focal coord mm (aligned with +RA on image)
        yFocal: scalar or 1D array
           spherical y focal coord mm (aligned with +Dec on image)
        zFocal: scalar or 1D array
            spherical z focal coord mm
        """
        # convert xyz to sph sys
        thetaField, phiField = cart2Sph(xField, yField, zField)
        phiFocal = self._forwardDistort(phiField)
        # convert back to xyz coords
        xFocal = self.r * numpy.cos(numpy.radians(thetaField)) * numpy.sin(numpy.radians(180 - phiFocal))
        yFocal = self.r * numpy.sin(numpy.radians(thetaField)) * numpy.sin(numpy.radians(180 - phiFocal))
        zFocal = self.r * numpy.cos(numpy.radians(180 - phiFocal)) + self.b
        # print(zFocal[0], zFocal[0]-self.b)
        return xFocal, yFocal, zFocal

    def focalToField(self, xFocal, yFocal, zFocal):
        """Convert xyz focal position to unit-spherical field position

        Parameters
        -----------
        xFocal: scalar or 1D array
            x focal coord mm (aligned with +RA on image)
        yFocal: scalar or 1D array
            y focal coord mm (aligned with +Dec on image)
        zFocal: scalar or 1D array
            z focal coord mm, +z points along boresight toward sky.

        Result
        -------
        xField: scalar or 1D array
            spherical x focal coord (aligned with +RA on sky)
        yField: scalar or 1D array
           spherical y focal coord (aligned with +Dec on sky)
        zField: scalar or 1D array
            spherical z coord
        """

        # note by definition thetaField==thetaFocal
        thetaField = numpy.degrees(numpy.arctan2(yFocal, xFocal))

        # generate focal phis (degree off boresight)
        # angle off-axis from optical axis
        rFocal = numpy.sqrt(xFocal**2 + yFocal**2)
        v = numpy.array([rFocal, zFocal]).T
        v[:, 1] = v[:, 1] - self.b

        # unit vector pointing toward object on focal plane from circle center
        # arc from vertex to off axis
        v = v / numpy.vstack([numpy.linalg.norm(v, axis=1)] * 2).T
        downwardZaxis = numpy.array([0, -1])  # FP lands at -Z so, Angle from sphere center towards ground
        # phi angle between ray and optical axis measured from sphere center
        phiFocal = numpy.degrees(numpy.arccos(v.dot(downwardZaxis)))
        phiField = self._reverseDistort(phiFocal)

        # finally report in unit-spherical xyz
        return sph2Cart(thetaField, phiField)


LCO_POWERS = [1, 3, 5, 7]
APO_POWERS = [1, 3, 5, 7, 9]

LCO_APOGEE_FOR_COEFFS = [2.11890e+00, 1.40826e-02, 1.27996e-04, 6.99967e-05]
LCO_APOGEE_REV_COEFFS = [4.71943e-01, -6.98482e-04, 1.58969e-06, -1.47239e-07]

LCO_BOSS_FOR_COEFFS = [1.89824e+00, 1.31773e-02, 1.04445e-04, 5.77341e-05]
LCO_BOSS_REV_COEFFS = [5.26803e-01, -1.01471e-03, 3.47109e-06, -2.98113e-07]

LCO_GFA_FOR_COEFFS = [1.93618e+00, 1.33536e-02, 9.17031e-05, 6.58945e-05]
LCO_GFA_REV_COEFFS = [ 5.16480e-01, -9.50007e-04, 3.34034e-06, -2.93032e-07]

APO_APOGEE_FOR_COEFFS = [1.40708e+00, 6.13779e-03, 7.25138e-04, -3.28007e-06, -1.65995e-05]
APO_APOGEE_REV_COEFFS = [7.10691e-01, -1.56306e-03, -8.60362e-05, 3.10036e-06, 3.16259e-07]

APO_BOSS_FOR_COEFFS = [1.36580e+00, 6.09425e-03, 6.54926e-04, 2.62176e-05, -2.27106e-05]
APO_BOSS_REV_COEFFS = [7.32171e-01, -1.74740e-03, -9.28511e-05, 1.80969e-06, 6.48944e-07]

APO_GFA_FOR_COEFFS = [1.37239e+00, 6.09825e-03, 6.67511e-04, 2.14437e-05, -2.17330e-05]
APO_GFA_REV_COEFFS = [7.28655e-01, -1.71534e-03, -9.19802e-05, 2.07648e-06, 5.84442e-07]

lcoApogeeModel = FocalPlaneModel(
    r=8905,
    b=7912,
    powers=LCO_POWERS,
    forwardCoeffs=LCO_APOGEE_FOR_COEFFS,
    reverseCoeffs=LCO_APOGEE_REV_COEFFS,
)

lcoBossModel = FocalPlaneModel(
    r=9938,
    b=8945,
    powers=LCO_POWERS,
    forwardCoeffs=LCO_BOSS_FOR_COEFFS,
    reverseCoeffs=LCO_BOSS_REV_COEFFS,
)

lcoGFAModel = FocalPlaneModel(
    r=9743,
    b=8751,
    powers=LCO_POWERS,
    forwardCoeffs=LCO_GFA_FOR_COEFFS,
    reverseCoeffs=LCO_GFA_REV_COEFFS,
)

apoApogeeModel = FocalPlaneModel(
    r=8939,
    b=8163,
    powers=APO_POWERS,
    forwardCoeffs=APO_APOGEE_FOR_COEFFS,
    reverseCoeffs=APO_APOGEE_REV_COEFFS,
)

apoBossModel = FocalPlaneModel(
    r=9208,
    b=8432,
    powers=APO_POWERS,
    forwardCoeffs=APO_BOSS_FOR_COEFFS,
    reverseCoeffs=APO_BOSS_REV_COEFFS,
)

apoGFAModel = FocalPlaneModel(
    r=9164,
    b=8388,
    powers=APO_POWERS,
    forwardCoeffs=APO_GFA_FOR_COEFFS,
    reverseCoeffs=APO_GFA_REV_COEFFS,
)

focalPlaneModelDict = {}
focalPlaneModelDict["LCO"] = {
    "Apogee": lcoApogeeModel,
    "Boss": lcoBossModel,
    "GFA": lcoGFAModel,
}
focalPlaneModelDict["APO"] = {
    "Apogee": apoApogeeModel,
    "Boss": apoBossModel,
    "GFA": apoGFAModel,
}


def fieldToFocal(x, y, z, observatory, waveCat):
    """Convert unit-spherical field coordinates to a position on
    a spherical focal plane.

    For each focalplane at each wavelength a spherical focal plane model
    with a polynomial distortion model is fit, this fitting is done
    in focalSurfaceModel.generateFits()

    The telescope roatates the input field by 180 degrees.  Field x
    is aligned with +RA, Field y is algned with +Dec, Field y points from
    the telescope to the sky.

    Focal x is aligned with +RA on the image (-RA on the sky)
    Focal y is aligned with +Dec on the image (-Dec on the sky)
    Focal z is aligned with the boresight, increasing from the telescope
        toward the sky

    The origin of the focal coordinate system is the M1 mirror vertex.

    Parameters
    -----------
    x: scalar or 1D array
        unit-spherical x field coord (aligned with +RA)
    y: scalar or 1D array
        unit-spherical y field coord (aligned with +Dec)
    z: scalar or 1D array
        unit-spherical z coord
    observatory: string
        either "APO" or "LCO"
    waveCat: string
        wavelength either "Apogee", "Boss", or "GFA"

    Returns
    --------
    x: scalar or 1D array
        x position of object on spherical focal plane mm
        (+x aligned with +RA on image)
    y: scalar or 1D array
        y position of object on spherical focal plane mm
        (+y aligned with +Dec on image)
    z: scalar or 1D array
        z position of object on spherical focal plane mm
        (+z aligned boresight and increases from the telescope to the sky)
    """
    # these paramters are obtained from fits by focalSurfaceModel.generateFits()
    if observatory not in ["APO", "LCO"]:
        raise RuntimeError("observatory must be APO or LCO")
    if waveCat not in ["Apogee", "Boss", "GFA"]:
        raise RuntimeError("waveCat must be one of Apogee, Boss, or GFA")
    model = focalPlaneModelDict[observatory][waveCat]
    return model.fieldToFocal(x,y,z)


def focalToField(x, y, z, observatory, waveCat):
    """Convert xyz focal coordinates to a unit-spherical xyz field position.

    For each focalplane at each wavelength a spherical focal plane model
    with a polynomial distortion model is fit, this fitting is done
    in focalSurfaceModel.generateFits()

    Focal x is aligned with +RA on the image (-RA on the sky)
    Focal y is aligned with +Dec on the image (-Dec on the sky)
    Focal z is aligned with the boresight, increasing from the telescope
        toward the sky

    The origin of the focal coordinate system is the M1 mirror vertex.

    Parameters
    -----------
    x: scalar or 1D array
        x position of object on focal plane mm
        (+x aligned with +RA on image)
    y: scalar or 1D array
        y position of object on focal plane mm
        (+y aligned with +Dec on image)
    z: scalar or 1D array
        z position of object on focal plane mm
        (+z aligned boresight and increases from the telescope to the sky)
    observatory: string
        either "APO" or "LCO"
    waveCat: string
        wavelength either "Apogee", "Boss", or "GFA"

    Returns
    --------
    x: scalar or 1D array
        unit-spherical x field coord (aligned with +RA on sky)
    y: scalar or 1D array
        unit-spherical y field coord (aligned with +Dec on sky)
    z: scalar or 1D array
        unit-spherical z coord
    """
    # these paramters are obtained from fits by focalSurfaceModel.generateFits()
    if observatory not in ["APO", "LCO"]:
        raise RuntimeError("observatory must be APO or LCO")
    if waveCat not in ["Apogee", "Boss", "GFA"]:
        raise RuntimeError("waveCat must be one of Apogee, Boss, or GFA")
    model = focalPlaneModelDict[observatory][waveCat]
    return model.focalToField(x,y,z)

#######
# these are calibrated parameters, they
# should live in a file somewhere and may
# change throughout time
# they are defined at PA=0!

# z offset is the distance between the M1 vertex
# and the wok vertex.  This measurement doesn't really
# exist in a model anywhere
POSITIONER_HEIGHT = 143 # 143.03 # mm (height of fiber above wok surface measured by kal)

# estimate the z offset by where the average focus position is
# on axis between the two focal planes
# a better way to do this is find out where stars
# come into focus on the GFAs
APO_WOK_Z_OFFSET = numpy.mean(
    [apoApogeeModel.b - apoApogeeModel.r,
    apoBossModel.b - apoBossModel.r
    ]
) - POSITIONER_HEIGHT

LCO_WOK_Z_OFFSET = numpy.mean(
    [lcoApogeeModel.b - lcoApogeeModel.r,
    lcoBossModel.b - lcoBossModel.r
    ]
) - POSITIONER_HEIGHT

# tranlational de-center of wok with focal plane
# the pointing model may handle this stuff
# APO_WOK_X_OFFSET = 0
# APO_WOK_Y_OFFSET = 0

# LCO_WOK_X_OFFSET = 0
# LCO_WOK_Y_OFFSET = 0

# # tilts of woks with respect to optical axis
# APO_WOK_TILT_X = 0 # tilt in degrees about x axis
# APO_WOK_TILT_Y = 0 # tilt in degrees about y axis

# LCO_WOK_TILT_X = 0 # tilt in degrees about x axis
# LCO_WOK_TILT_Y = 0 # tilt in degrees about y axis
#######

# print("z offsets", APO_WOK_Z_Offset, LCO_WOK_Z_Offset)


def focalToWok(
    xFocal, yFocal, zFocal, positionAngle=0,
    xOffset=0, yOffset=0, zOffset=0, tiltX=0, tiltY=0
):
    """Convert xyz focal coordinates in mm to xyz wok coordinates in mm.

    The origin of the focal coordinate system is the
    M1 vertex. focal +y points toward North, +x points toward E.
    The origin of the wok coordinate system is the wok vertex.  -x points
    toward the boss slithead.  +z points from telescope to sky.

    Tilt is applied about x axis then y axis.


    Parameters
    -------------
    xFocal: scalar or 1D array
        x position of object on focal plane mm
        (+x aligned with +RA on image)
    yFocal: scalar or 1D array
        y position of object on focal plane mm
        (+y aligned with +Dec on image)
    zFocal: scalar or 1D array
        z position of object on focal plane mm
        (+z aligned boresight and increases from the telescope to the sky)
    positionAngle: scalar
        position angle deg.  Angle measured from (image) North through East to wok +y.
        So position angle of 45 deg, wok +y points NE
    xOffset: scalar or None
        x position (mm) of wok origin (vertex) in focal coords
        calibrated
    yOffset: scalar
        y position (mm) of wok origin (vertex) in focal coords
        calibratated
    zOffset: scalar
        z position (mm) of wok origin (vertex) in focal coords
        calibratated
    tiltX: scalar
        tilt (deg) of wok about focal x axis at PA=0
        calibrated
    tiltY: scalar
        tilt (deg) of wok about focal y axis at PA=0
        calibrated

    Returns
    ---------
    xWok: scalar or 1D array
        x position of object in wok space mm
        (+x aligned with +RA on image)
    yWok: scalar or 1D array
        y position of object in wok space mm
        (+y aligned with +Dec on image)
    zWok: scalar or 1D array
        z position of object in wok space mm
        (+z aligned boresight and increases from the telescope to the sky)
    """

    # apply calibrated tilts and translation (at PA=0)
    # where they should be fit
    # tilts are defined https://mathworld.wolfram.com/RotationMatrix.html
    # as coordinate system rotations counter clockwise when looking
    # down the positive axis toward the origin
    coords = numpy.array([xFocal, yFocal, zFocal])
    transXYZ = numpy.array([xOffset, yOffset, zOffset])

    rotX = numpy.radians(tiltX)
    rotY = numpy.radians(tiltY)
    # rotation about z axis is position angle
    # position angle is clockwise positive for rotation measured from
    # north to wok +y (when looking from above the wok)
    # however rotation matrices are positinve for counter-clockwise rotation
    # hence the sign flip that's coming
    rotZ = -1*numpy.radians(positionAngle)

    rotMatX = numpy.array([
        [1, 0, 0],
        [0, numpy.cos(rotX), numpy.sin(rotX)],
        [0, -1*numpy.sin(rotX), numpy.cos(rotX)]
    ])

    rotMatY = numpy.array([
        [numpy.cos(rotY), 0, -1*numpy.sin(rotY)],
        [0, 1, 0],
        [numpy.sin(rotY), 0, numpy.cos(rotY)]
    ])

    # rotates coordinate system
    rotMatZ = numpy.array([
        [numpy.cos(rotZ), numpy.sin(rotZ), 0],
        [-numpy.sin(rotZ), numpy.cos(rotZ), 0],
        [0, 0, 1]
    ])


    # first apply rotation about x axis
    coords = rotMatX.dot(coords)
    # next apply rotation about y axis
    coords = rotMatY.dot(coords)
    # apply rotation about z axis (PA)
    coords = rotMatZ.dot(coords)


    print("coords", coords)

    # apply translation
    if hasattr(xFocal, "__len__"):
        # list of coords fed in
        transXYZ = numpy.array([transXYZ]*len(xFocal)).T
        xWok, yWok, zWok = coords - transXYZ
    else:
        # single set of xyz coords fed in
        xWok, yWok, zWok = coords - transXYZ

    # # rotate about z axis (PA setting)
    # xWok, yWok, zWok = rotMatZ.dot(coords)

    return xWok, yWok, zWok


def wokToFocal(
    xWok, yWok, zWok, positionAngle=0,
    xOffset=0, yOffset=0, zOffset=0, tiltX=0, tiltY=0
):
    """Convert xyz wok coordinates in mm to xyz focal coordinates in mm.

    The origin of the focal coordinate system is the
    M1 vertex. focal +y points toward North, +x points toward E.
    The origin of the wok coordinate system is the wok vertex.  -x points
    toward the boss slithead.  +z points from telescope to sky.

    Tilt is applied about x axis then y axis.


    Parameters
    -------------
    xWok: scalar or 1D array
        x position of object in wok space mm
        (+x aligned with +RA on image)
    yWok: scalar or 1D array
        y position of object in wok space mm
        (+y aligned with +Dec on image)
    zWok: scalar or 1D array
        z position of object in wok space mm
        (+z aligned boresight and increases from the telescope to the sky)
    positionAngle: scalar
        position angle deg.  Angle measured from (image) North through East to wok +y.
        So position angle of 45 deg, wok +y points NE
    xOffset: scalar or None
        x position (mm) of wok origin (vertex) in focal coords
        calibrated
    yOffset: scalar
        y position (mm) of wok origin (vertex) in focal coords
        calibratated
    zOffset: scalar
        z position (mm) of wok origin (vertex) in focal coords
        calibratated
    tiltX: scalar
        tilt (deg) of wok about focal x axis at PA=0
        calibrated
    tiltY: scalar
        tilt (deg) of wok about focal y axis at PA=0
        calibrated

    Returns
    ---------
    xFocal: scalar or 1D array
        x position of object in focal coord sys mm
        (+x aligned with +RA on image)
    yFocal: scalar or 1D array
        y position of object in focal coord sys mm
        (+y aligned with +Dec on image)
    zFocal: scalar or 1D array
        z position of object in focal coord sys mm
        (+z aligned boresight and increases from the telescope to the sky)
    """
    # this routine is a reversal of the steps
    # in the function focalToWok, with rotational
    # angles inverted and translations applied in reverse
    coords = numpy.array([xWok, yWok, zWok])

    rotX = numpy.radians(-1*tiltX)
    rotY = numpy.radians(-1*tiltY)
    rotZ = numpy.radians(positionAngle)

    rotMatX = numpy.array([
        [1, 0, 0],
        [0, numpy.cos(rotX), numpy.sin(rotX)],
        [0, -numpy.sin(rotX), numpy.cos(rotX)]
    ])

    rotMatY = numpy.array([
        [numpy.cos(rotY), 0, -numpy.sin(rotY)],
        [0, 1, 0],
        [numpy.sin(rotY), 0, numpy.cos(rotY)]
    ])

    rotMatZ = numpy.array([
        [numpy.cos(rotZ), numpy.sin(rotZ), 0],
        [-numpy.sin(rotZ), numpy.cos(rotZ), 0],
        [0, 0, 1]
    ])

    transXYZ = numpy.array([xOffset, yOffset, zOffset])

    # add offsets for reverse transform
    if hasattr(xWok, "__len__"):
        # list of coords fed in
        transXYZ = numpy.array([transXYZ]*len(xWok)).T
        coords = coords + transXYZ
    else:
        # single set of xyz coords fed in
        coords = coords + transXYZ

    coords = rotMatZ.dot(coords)
    coords = rotMatY.dot(coords)
    xFocal, yFocal, zFocal = rotMatX.dot(coords)
    return xFocal, yFocal, zFocal


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    numpy.random.seed(1)
    azCen = 180+35 # degrees, roughly SW
    altCen = 45 # degrees
    latitude = 35
    azCoords = azCen + numpy.random.uniform(-1,1, size=30)
    altCoords = altCen + numpy.random.uniform(-1,1, size=30)
    azAlt = numpy.c_[azCoords, altCoords]
    # add last coord as center
    azAlt[-1] = numpy.array([azCen, altCen])
    print(azAlt.shape)
    plt.figure()
    plt.plot(azAlt[:,0], azAlt[:,1], 'o')
    plt.plot(azAlt[-1,0], azAlt[-1,1], 'x')
    plt.xlabel("Az")
    plt.ylabel("Alt")
    # plt.show()


    # # convert azAlt to unit sphere x=North, y=West
    # thetas=azAlt[:,0]*-1
    # phis = 90 - azAlt[:,1]
    # carts = []
    # for theta, phi in zip(thetas, phis):
    #     # x = south
    #     # y = east
    #     # z = zenith
    #     carts.append(sph2Cart(theta, phi))
    # carts = numpy.array(carts)

    # # rotate about z axes such that field center lies on -y
    # sinT1 = numpy.sin(numpy.radians(90 + azCen))
    # cosT1 = numpy.cos(numpy.radians(90 + azCen))
    # rot1 = numpy.array([
    #     [ cosT1, sinT1, 0],
    #     [-sinT1, cosT1, 0],
    #     [     0,     0, 1]
    # ])
    # carts2 = rot1.dot(carts.T).T
    # # carts2 = carts.dot(rot1.T)
    # plt.figure()
    # # image flipped left right due to looking from
    # # outside sphere now...
    # plt.plot(carts2[:,0]*-1, carts2[:,2], 'o') # plot x vs z
    # plt.plot(carts2[-1,0]*-1, carts2[-1,2], 'x') # field center
    # plt.xlabel("-x")
    # plt.ylabel("z")

    # # rotate about y axis such that field center
    # # is aligned with +z
    # sinT2 = numpy.sin(numpy.radians(90-altCen))
    # cosT2 = numpy.cos(numpy.radians(90-altCen))
    # rot2 = numpy.array([
    #     [1, 0, 0],
    #     [0,  cosT2, sinT2],
    #     [0, -sinT2, cosT2]
    # ])

    # carts3 = rot2.dot(carts2.T).T
    # plt.figure()
    # plt.plot(carts3[:,0]*-1, carts3[:,1], 'o') # plot x vs y
    # plt.plot(carts3[-1,0]*-1, carts3[-1,1], 'x') # plot x vs y
    # plt.xlabel("-x")
    # plt.ylabel("y")

    x, y, z = observedToField(azAlt[:,0], azAlt[:,1], azCen, altCen, latitude)
    plt.figure()
    plt.plot(x*-1, y, 'o') # plot x vs y
    plt.plot(x[-1]*-1, y[-1], 'x') # plot x vs y
    plt.xlabel("-x")
    plt.ylabel("y")

    x,y,z = observedToField(azAlt[1,0], azAlt[1,1], azCen, altCen, latitude)
    plt.plot(x*-1, y, '+r')

    # import pdb; pdb.set_trace()

    # lat = 45
    # azs = numpy.random.uniform(0, 360, size=10000)
    # alts = numpy.random.uniform(0, 90, size=10000)
    # d = {}
    # d["az"] = azs
    # d["alt"] = alts
    # ha, dec = azAlt2HaDec(azs, alts, lat)
    # d["ha"] = ha
    # d["dec"] = dec
    # q = parallacticAngle(ha, dec, lat)
    # print("q", q)
    # plt.figure()
    # plt.hist(q)
    # d["q"] = q
    # d = pd.DataFrame(d)
    # plt.figure()
    # sns.scatterplot(x="az", y="alt", hue="q", data=d)
    # plt.figure()
    # sns.scatterplot(x="ha", y="dec", hue="q", data=d)

    plt.show()









