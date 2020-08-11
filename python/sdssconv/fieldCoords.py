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
    """Convert cartesian coordinates on unit sphere to spherical
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


def sph2Cart(theta, phi):
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

    Returns
    ---------
    result: list
        [x,y,z] coordinates on unit sphere

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
    x = numpy.cos(theta) * numpy.sin(phi)
    y = numpy.sin(theta) * numpy.sin(phi)
    z = numpy.cos(phi)

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


def azAlt2Cart(az, alt, azCenter, altCenter, latitude):
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



# def cart2AzAlt(x, y, z, azAltCen, latitude):
#     """Convert field xyz coordinates on the unit sphere to azAlt coords

#     Parameters
#     ------------
#     cartXYZ: numpy.array
#         [x,y,z] coordinates on unit sphere +x aligned with +RA, +y
#         aligned with +DEC
#     azAltCen: array
#         [azimuth, altitude] coordinates of field center
#     latitude: float
#         observer latitude in degrees, positive for north

#     Returns
#     --------
#     result: array
#         [azimuth, altitude] coordinates in degrees.
#         Az=0 south, Az=90 equals west

#     """
#     pass


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

    x, y, z = azAlt2Cart(azAlt[:,0], azAlt[:,1], azCen, altCen, latitude)
    plt.figure()
    plt.plot(x*-1, y, 'o') # plot x vs y
    plt.plot(x[-1]*-1, y[-1], 'x') # plot x vs y
    plt.xlabel("-x")
    plt.ylabel("y")

    x,y,z = azAlt2Cart(azAlt[1,0], azAlt[1,1], azCen, altCen, latitude)
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









