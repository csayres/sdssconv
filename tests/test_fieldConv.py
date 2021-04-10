import pytest
import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sdssconv.fieldCoords import cart2FieldAngle, fieldAngle2Cart
from sdssconv.fieldCoords import sph2Cart, cart2Sph, SMALL_NUM
from sdssconv.fieldCoords import observedToField, fieldToObserved
from sdssconv.fieldCoords import focalToField, fieldToFocal, focalPlaneModelDict
from sdssconv.fieldCoords import focalToWok, wokToFocal, APO_WOK_Z_OFFSET, LCO_WOK_Z_OFFSET
from sdssconv.fieldCoords import wokToTangent, tangentToWok
from sdssconv.fieldCoords import proj2XYplane, tangentToGuide, guideToTangent, PIXEL_SIZE, CHIP_CENTER
from sdssconv.fieldCoords import tangentToPositioner, positionerToTangent

numpy.random.seed(0)

APO_latitude = 32.7802
LCO_latitude = -29.0182
nPts = 100000
r = 1 # unit sphere
thetas = numpy.random.uniform(size=nPts) * 2 * numpy.pi
# maximum of 4 degrees off axis
phis = numpy.radians(numpy.random.uniform(size=nPts) * 4)
xs = r * numpy.cos(thetas) * numpy.sin(phis)
ys = r * numpy.sin(thetas) * numpy.sin(phis)
zs = r * numpy.cos(phis)


def test_phiConversion(verbose=False):
    for fieldAngle in numpy.linspace(-10,10,100):
        x,y,z = fieldAngle2Cart(fieldAngle, 0)
        theta, phi = cart2Sph(x,y,z)
        if verbose:
            print(numpy.abs(phi-fieldAngle))
        else:
            assert numpy.abs(phi-numpy.abs(fieldAngle)) < SMALL_NUM

        x,y,z = fieldAngle2Cart(0, fieldAngle)
        theta, phi = cart2Sph(x,y,z)
        if verbose:
            print(numpy.abs(phi-fieldAngle))
        else:
            assert numpy.abs(phi-numpy.abs(fieldAngle)) < SMALL_NUM


def test_cartField():
    # pick some random points on the unit sphere near the +Z cap
    for theta, phi, x, y, z in zip(thetas, phis, xs, ys, zs):
        xField, yField = cart2FieldAngle(x, y, z)
        if theta < numpy.pi / 2 or theta > 3 * numpy.pi / 2:
            assert xField < 0
        else:
            assert xField > 0

        if theta < numpy.pi:
            assert yField < 0
        else:
            assert yField > 0

        xSolve, ySolve, zSolve = fieldAngle2Cart(xField, yField)
        # print(xSolve - x, ySolve - y, zSolve - z)

        assert numpy.abs(xSolve-x) < SMALL_NUM
        assert numpy.abs(ySolve-y) < SMALL_NUM
        assert numpy.abs(zSolve-z) < SMALL_NUM


def test_cartFieldCycle():
    # check round trippage
    inds = numpy.random.choice(range(nPts), size=100)
    for ind in inds:
        x = xs[ind]
        y = ys[ind]
        z = zs[ind]
        _x = xs[ind]
        _y = ys[ind]
        _z = zs[ind]
        for ii in range(100):
            # print("ii", ii)
            xField, yField = cart2FieldAngle(_x,_y,_z)
            _x, _y, _z = fieldAngle2Cart(xField, yField)
            # repeated round trips require extra numerical buffer
            assert numpy.abs(_x-x) < SMALL_NUM
            assert numpy.abs(_y-y) < SMALL_NUM
            assert numpy.abs(_z-z) < SMALL_NUM


def test_sphCartCycle():
    # check round trippage
    inds = numpy.random.choice(range(nPts), size=100)
    for ind in inds:
        x = xs[ind]
        y = ys[ind]
        z = zs[ind]
        theta = numpy.degrees(thetas[ind])
        phi = numpy.degrees(phis[ind])
        _x = xs[ind]
        _y = ys[ind]
        _z = zs[ind]
        for ii in range(100):
            _theta, _phi = cart2Sph(_x, _y, _z)
            # print("[%.4f, %.4f] == %.5e, %.5e"%(theta, phi, phiTheta[0]-theta, phiTheta[1]-phi))
            assert numpy.abs(_theta - theta) < SMALL_NUM
            assert numpy.abs(_phi - phi) < SMALL_NUM
            _x, _y, _z = sph2Cart(_theta, _phi)
            assert numpy.abs(_x-x) < SMALL_NUM
            assert numpy.abs(_y-y) < SMALL_NUM
            assert numpy.abs(_z-z) < SMALL_NUM


def test_observedToField_APO():

    #### start checks along the meridian
    ####################################
    azCen = 180 # (south)
    altCen = 45
    azObj = azCen
    altObj = altCen + 1 # north of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, APO_latitude)
    assert abs(x) < SMALL_NUM
    assert y > 0

    azCen = 0 # (north)
    altCen = 45 # above north star
    azObj = azCen
    altObj = altCen + 1 # south of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, APO_latitude)
    assert abs(x) < SMALL_NUM
    assert y < 0

    azCen = 0 # (north)
    altCen = 20 # below north star
    azObj = azCen
    altObj = altCen + 1 # north of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, APO_latitude)
    assert abs(x) < SMALL_NUM
    assert y > 0

    ##### check field rotation (off meridian)
    #########################################
    # remember +x is eastward
    azCen = 180 + 20 # (south-west)
    altCen = 45
    azObj = azCen
    altObj = altCen + 1 # north-east of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, APO_latitude)
    assert x > 0
    assert y > 0

    # check field rotation (off meridian)
    # remember +x is eastward
    azCen = 180 - 20 # (south-east)
    altCen = 45
    azObj = azCen
    altObj = altCen + 1 # north-east of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, APO_latitude)
    assert x < 0
    assert y > 0

    # check field rotation (off meridian)
    # remember +x is eastward
    azCen = 10 # (slightly east of north)
    altCen = APO_latitude - 10
    azObj = azCen
    altObj = altCen + 1 # north-east of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, APO_latitude)
    assert x < 0
    assert y > 0

    azCen = 10 # (slightly east of north)
    altCen = APO_latitude + 10
    azObj = azCen
    altObj = altCen + 1 # north-east of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, APO_latitude)
    assert x < 0
    assert y < 0


def test_observedToField_LCO():

    #### start checks along the meridian
    ####################################
    azCen = 0 # (north)
    altCen = 45
    azObj = azCen
    altObj = altCen + 1 # south of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, LCO_latitude)
    assert abs(x) < SMALL_NUM
    assert y < 0

    azCen = 180 # (south)
    altCen = numpy.abs(LCO_latitude) + 10 # above SCP, (lat is negative!)
    azObj = azCen
    altObj = altCen + 1 # north of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, LCO_latitude)
    assert abs(x) < SMALL_NUM
    assert y > 0

    azCen = 180 # (south)
    altCen = numpy.abs(LCO_latitude) - 10 # below SCP, (lat is negative!)
    azObj = azCen
    altObj = altCen + 1 # north of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, LCO_latitude)
    assert abs(x) < SMALL_NUM
    assert y < 0

    ##### check field rotation (off meridian)
    #########################################
    # remember +x is eastward
    azCen = 20 # (north-east)
    altCen = 45
    azObj = azCen
    altObj = altCen + 1 # north-east of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, LCO_latitude)
    assert x < 0
    assert y < 0

    # remember +x is eastward
    azCen = 290 # (north-west)
    altCen = 45
    azObj = azCen
    altObj = altCen + 1 # north-east of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, LCO_latitude)
    assert x > 0
    assert y < 0

    # # check field rotation (off meridian)
    # # remember +x is eastward
    azCen = 170 # (slightly east of south)
    altCen = LCO_latitude - 10
    azObj = azCen
    altObj = altCen + 1 # north-east of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, LCO_latitude)
    assert x < 0
    assert y < 0

    azCen = 190 # (slightly west of south)
    altCen = LCO_latitude - 10
    azObj = azCen
    altObj = altCen + 1 # north-east of field center
    x, y, z = observedToField(azObj, altObj, azCen, altCen, LCO_latitude)
    assert x > 0
    assert y < 0


def test_observedToFieldCycles():
    # try a bunch of pointings make sure the round trip works
    azCens = numpy.random.uniform(0,360, size=30)
    altCens = numpy.random.uniform(0,90, size=30)
    for azCen, altCen in zip(azCens, altCens):
        azCoords = azCen + numpy.random.uniform(-1,1, size=30)
        altCoords = altCen + numpy.random.uniform(-1,1, size=30)
        x, y, z = observedToField(azCoords, altCoords, azCen, altCen, APO_latitude)
        _azCoords, _altCoords = fieldToObserved(x, y, z, azCen, altCen, APO_latitude)
        assert numpy.max(numpy.abs(azCoords-_azCoords)) < SMALL_NUM
        assert numpy.max(numpy.abs(altCoords-_altCoords)) < SMALL_NUM


def test_fieldToFocal():
    # test zero field
    for obs in ["APO", "LCO"]:
        for waveCat in ["Apogee", "Boss", "GFA"]:
            x,y,z = 0,0,1
            # check on axis
            _x,_y,_z = fieldToFocal(x,y,z,obs,waveCat)
            assert numpy.abs(_x) < SMALL_NUM
            assert numpy.abs(_y) < SMALL_NUM
            # check array on axis
            x = [0,0,0]
            y = [0,0,0]
            z = [1,1,1]
            _x,_y,_z = fieldToFocal(x,y,z,obs,waveCat)
            assert numpy.max(numpy.abs(_x)) < SMALL_NUM
            assert numpy.max(numpy.abs(_y)) < SMALL_NUM

            theta = 0
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert x > 0
            assert numpy.abs(y) < SMALL_NUM
            _x,_y,_z = fieldToFocal(x,y,z,obs,waveCat)
            assert _x > 0
            assert numpy.abs(_y) < SMALL_NUM
            assert _z < 0

            theta = 90
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert numpy.abs(x) < SMALL_NUM
            assert y > 0
            assert z > 0
            _x,_y,_z = fieldToFocal(x,y,z,obs,waveCat)
            assert _y > 0
            assert numpy.abs(_x) < SMALL_NUM
            assert _z < 0

            theta = 180
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert numpy.abs(y) < SMALL_NUM
            assert x < 0
            assert z > 0
            _x,_y,_z = fieldToFocal(x,y,z,obs,waveCat)
            assert _x < 0
            assert numpy.abs(_y) < SMALL_NUM
            assert _z < 0

            theta = 270
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert numpy.abs(x) < SMALL_NUM
            assert y < 0
            assert z > 0
            _x,_y,_z = fieldToFocal(x,y,z,obs,waveCat)
            assert _y < 0
            assert numpy.abs(_x) < SMALL_NUM
            assert _z < 0

            theta = 300
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert x > 0
            assert y < 0
            assert z > 0
            _x,_y,_z = fieldToFocal(x,y,z,obs,waveCat)
            assert _y < 0
            assert _x > 0
            assert _z < 0

            theta = 380
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert x > 0
            assert y > 0
            assert z > 0
            _x,_y,_z = fieldToFocal(x,y,z,obs,waveCat)
            assert _y > 0
            assert _x > 0
            assert _z < 0

            theta = -20
            phi = 0.5 # 0.5 degrees off axis
            x,y,z = sph2Cart(theta, phi)
            assert x > 0
            assert y < 0
            assert z > 0
            _x,_y,_z = fieldToFocal(x,y,z,obs,waveCat)
            assert _y < 0
            assert _x > 0
            assert _z < 0


def test_focalCurvatureRadii():
    APO_max_field = 1.5
    LCO_max_field = 1
    for waveCat in ["Apogee", "Boss", "GFA"]:
        for obs, maxField in zip(["APO", "LCO"], [APO_max_field, LCO_max_field]):
            thetas = numpy.random.uniform(0,360,size=1000)
            phis = numpy.random.uniform(0,maxField,size=1000)
            x,y,z = sph2Cart(thetas, phis)
            fxyz = numpy.array(fieldToFocal(x,y,z,obs,waveCat)).T
            fxyz[:,2] = fxyz[:,2] + focalPlaneModelDict[obs][waveCat].b # b is z offset to origin
            radii = numpy.linalg.norm(fxyz, axis=1)
            expectedRadii = focalPlaneModelDict[obs][waveCat].r
            assert numpy.max(numpy.abs(radii-expectedRadii))


def test_fieldFocalCycle():
    APO_max_field = 1.5
    LCO_max_field = 1
    for waveCat in ["Apogee", "Boss", "GFA"]:
        for obs, maxField in zip(["APO", "LCO"], [APO_max_field, LCO_max_field]):
            thetas = numpy.random.uniform(0,360,size=10000)
            phis = numpy.random.uniform(0,maxField,size=10000)
            x,y,z = sph2Cart(thetas, phis)
            mag = numpy.linalg.norm(numpy.array([x,y,z]), axis=0)
            assert numpy.sum(numpy.abs(mag-1)) < SMALL_NUM
            fx,fy,fz = fieldToFocal(x,y,z,obs,waveCat)
            _x, _y, _z = focalToField(fx,fy,fz,obs,waveCat)
            # make sure unit spherical
            mag = numpy.linalg.norm(numpy.array([_x,_y,_z]), axis=0)
            assert numpy.sum(numpy.abs(mag-1)) < SMALL_NUM
            # convert back to spherical
            _thetas, _phis = cart2Sph(_x,_y,_z)
            assert numpy.max(numpy.abs(thetas-_thetas)) < SMALL_NUM
            # angular separation in arcseconds from round trip
            # from dot product formula
            g = x*_x + y*_y + z*_z
            # numerical overflow makes g > 1 sometimes?
            # this screws up the arccosine below, so round it
            g = numpy.round(g,10)
            # print("min max", numpy.min(g), numpy.max(g))
            angSep = numpy.degrees(numpy.arccos(g))*3600
            assert numpy.max(angSep) < 0.0001 # arcsec, basically no error


def test_focalToWok():
    zOff = -100
    xOff = 0
    yOff = 0
    xTilt = 0
    yTilt = 0
    positionAngle = 0

    xFocal, yFocal, zFocal = 0, 0, 0
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt
    )

    assert xWok == 0
    assert yWok == 0
    assert zWok == -1*zOff

    positionAngle = 90 # +y wok aligned with +x FP

    xFocal, yFocal, zFocal = 10, 0, 0

    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt
    )
    assert yWok == xFocal
    assert numpy.abs(xWok) < SMALL_NUM

    positionAngle = -90 # +y wok aligned with -x FP

    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt
    )

    assert yWok == -1*xFocal
    assert numpy.abs(xWok) < SMALL_NUM

    # test translation
    xFocal, yFocal, zFocal = 0, 0, 0
    xOff = 10
    positionAngle = 0
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt
    )

    assert xWok == -1*xOff
    assert numpy.abs(yWok) < SMALL_NUM

    yOff = 10
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt
    )

    assert xWok == -1*xOff
    assert yWok == -1*yOff

    positionAngle = 45
    xOff, yOff = 10, 10
    xFocal, yFocal, zFocal = 0,0,0
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt
    )
    assert xWok == yWok
    assert xWok < 0

    positionAngle = 45
    xOff, yOff = -10, 10
    xFocal, yFocal, zFocal = 0,0,0
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt
    )
    assert xWok == -1*yWok
    assert xWok > 0

    b = 0.5*numpy.sqrt(2*10**2)
    a = numpy.sqrt(2*b**2)
    positionAngle = 45
    xOff, yOff, zOff = 10, 10, 0
    xTilt, yTilt = 0, 0
    xFocal, yFocal, zFocal = b, b, 0

    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt
    )
    assert numpy.abs(xWok + a) < SMALL_NUM
    assert numpy.abs(yWok) < SMALL_NUM

    # test tilts
    positionAngle =0
    xOff, yOff, zOff = 0, 0, 0
    xFocal, yFocal, zFocal = 1, 0, 0
    xTilt = 1
    yTilt = 0
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt
    )
    assert xWok == xFocal
    assert yWok == yFocal
    assert zWok == zFocal

    positionAngle =0
    xOff, yOff, zOff = 0, 0, 0
    xFocal, yFocal, zFocal = 0, 1, 0
    xTilt = 1
    yTilt = 0
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt
    )

    assert xWok == 0
    assert yWok < 1
    assert yWok > 0
    assert zWok < 0

    positionAngle =0
    xOff, yOff, zOff = 0, 0, 0
    xFocal, yFocal, zFocal = 1, 1, 0
    xTilt = 1
    yTilt = 4
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt
    )

    assert xWok > 0
    assert xWok < 1
    assert yWok < 1
    assert yWok > 0
    assert yWok < 1
    assert zWok > 0

    positionAngle =0
    xOff, yOff, zOff = 0, 0, 0
    xFocal, yFocal, zFocal = 1, 1, 0
    xTilt = 1
    yTilt = -1
    xWok, yWok, zWok = focalToWok(
        xFocal, yFocal, zFocal, positionAngle,
        xOff, yOff, zOff, xTilt, yTilt
    )

    assert xWok > 0
    assert xWok < 1
    assert yWok < 1
    assert yWok > 0
    assert yWok < 1
    assert zWok < 0



def test_focalWokCycle():
    nPts = 1000
    APO_max_field = 1.5
    LCO_max_field = 1
    for waveCat in ["Apogee", "Boss", "GFA"]:
        for obs, maxField, zOff in zip(["APO", "LCO"], [APO_max_field, LCO_max_field], [APO_WOK_Z_OFFSET, LCO_WOK_Z_OFFSET]):
            thetas = numpy.random.uniform(0,360,size=nPts)
            phis = numpy.random.uniform(0,maxField,size=nPts)
            for seed in numpy.arange(100):
                # try random calibrations
                xOffset = numpy.random.uniform(-10, 10)
                yOffset = numpy.random.uniform(-10, 10)
                xTilt = numpy.random.uniform(-2, 2)
                yTilt = numpy.random.uniform(-2, 2)
                x,y,z = sph2Cart(thetas, phis)
                fx,fy,fz = fieldToFocal(x,y,z,obs,waveCat)
                wx, wy, wz = focalToWok(fx,fy,fz, xOffset, yOffset, zOff, xTilt, yTilt)
                _fx, _fy, _fz = wokToFocal(wx, wy, wz, xOffset, yOffset, zOff, xTilt, yTilt)
                assert numpy.max(numpy.abs(fx-_fx)) < SMALL_NUM
                assert numpy.max(numpy.abs(fy-_fy)) < SMALL_NUM
                assert numpy.max(numpy.abs(fz-_fz)) < SMALL_NUM

def test_flatWok():
    # flat wok unit directions
    iHat=[1,0,0]
    jHat = [0,1,0]
    kHat = [0,0,1]


    b = [0,0,0] #wok vertex
    xWok = 1
    yWok = 1
    zWok = 0

    tx,ty,tz = wokToTangent(xWok, yWok, zWok, b, iHat, jHat, kHat)

    assert tx == xWok
    assert ty == yWok
    assert tz == -143 # origin is 143 mm above wok surface

    b = [1,1,0]
    xWok,yWok,zWok = 1,1,0
    tx,ty,tz = wokToTangent(xWok, yWok, zWok, b, iHat, jHat, kHat)
    assert tx == 0
    assert ty == 0
    assert tz == -143

    b = [1,1,0]
    xWok,yWok,zWok = 0,0,0
    tx,ty,tz = wokToTangent(xWok, yWok, zWok, b, iHat, jHat, kHat)
    assert tx == -1
    assert ty == -1
    assert tz == -143

    b = [0,0,0]
    xWok,yWok,zWok = 1,1,0
    dRot = 45
    tx,ty,tz = wokToTangent(xWok, yWok, zWok, b, iHat, jHat, kHat, dRot=dRot)

    assert numpy.abs(tx - numpy.sqrt(2)) < SMALL_NUM
    assert numpy.abs(ty) < SMALL_NUM
    assert tz == -143

    b = [2,2,0]
    xWok,yWok,zWok = 1,1,0
    dRot = 45
    tx,ty,tz = wokToTangent(xWok, yWok, zWok, b, iHat, jHat, kHat, dRot=dRot)

    assert numpy.abs(tx + numpy.sqrt(2)) < SMALL_NUM
    assert numpy.abs(ty) < SMALL_NUM
    assert tz == -143

def test_curvedWok():
    csvFile = os.getenv("SDSSCONV_DIR") +  "/data/wokCoords.csv"
    wokCoords = pd.read_csv(csvFile, index_col=0)

    for idx, row in wokCoords.iterrows():

        b = row[["x", "y", "z"]]
        iHat = row[["ix", "iy", "iz"]]
        jHat = row[["jx", "jy", "jz"]]
        kHat = row[["kx", "ky", "kz"]]
        xWok = b[0]
        yWok = b[1]
        zWok = b[2]
        tx,ty,tz = wokToTangent(xWok, yWok, zWok, b, iHat, jHat, kHat)
        assert numpy.sqrt(tx*2+ty**2) < SMALL_NUM
        assert (tz+143) < SMALL_NUM


def test_wokTangentCycle():
    csvFile = os.getenv("SDSSCONV_DIR") +  "/data/wokCoords.csv"
    wokCoords = pd.read_csv(csvFile, index_col=0)

    for idx, row in wokCoords.iterrows():
        b = row[["x", "y", "z"]]
        iHat = row[["ix", "iy", "iz"]]
        jHat = row[["jx", "jy", "jz"]]
        kHat = row[["kx", "ky", "kz"]]

        # import pdb; pdb.set_trace()
        for seed in range(10):
            tx = numpy.random.uniform(-50,50)
            ty = numpy.random.uniform(-50,50)
            tz = numpy.random.uniform(-50,50)
            scaleFac = numpy.random.uniform(0.8, 1.2)
            dx = numpy.random.uniform(-5, 5)
            dy = numpy.random.uniform(-5, 5)
            dz = numpy.random.uniform(-5, 5)
            dRot = numpy.random.uniform(-20,20)
            elementHeight = 143
            wx,wy,wz = tangentToWok(
                tx,ty,tz,b,iHat,jHat,kHat,
                elementHeight, scaleFac, dx, dy, dz, dRot
            )
            _tx, _ty, _tz = wokToTangent(
                wx, wy, wz, b,iHat,jHat,kHat,
                elementHeight, scaleFac, dx, dy, dz, dRot
            )
            norm = numpy.sqrt(
                (tx-_tx)**2 + (ty-_ty)**2 + (tz-_tz)**2
            )
            # print("norm", norm)
            assert norm < SMALL_NUM

def test_wokVsFocalSurface():
    csvFile = os.getenv("SDSSCONV_DIR") +  "/data/wokCoords.csv"
    wokCoords = pd.read_csv(csvFile, index_col=0)
    lcoCoords = wokCoords[wokCoords["wokType"]=="LCO"]
    apoCoords = wokCoords[wokCoords["wokType"]=="APO"]


    for cs in [lcoCoords, apoCoords]:
        bs = []
        fs = []
        for idx, row in cs.iterrows():
            b = row[["x", "y", "z"]].to_numpy()
            bs.append(b)
            iHat = row[["ix", "iy", "iz"]]
            jHat = row[["jx", "jy", "jz"]]
            kHat = row[["kx", "ky", "kz"]]
            f = tangentToWok(
                0, 0, 0, b, iHat, jHat, kHat,
            )
            fs.append(numpy.array(f))

        bs = numpy.array(bs, dtype="float64")
        fs = numpy.array(fs, dtype="float64")
        r1 = numpy.linalg.norm(bs[:,:2], axis=1)
        z1 = bs[:,2]
        r2 = numpy.linalg.norm(fs[:,:2], axis=1)
        z2 = fs[:,2]

        sortIdx = numpy.argsort(r1)
        r1 = r1[sortIdx]
        z1 = r1[sortIdx]
        r2 = r2[sortIdx]
        z2 = z2[sortIdx]

        # on axis no curve
        assert numpy.abs(r1[0]-r2[0]) < SMALL_NUM

        assert numpy.sum(r2[1:]>r1[1:]) == 0

        # could add a test to show that angle increases
        # with radius but dont have a good way to test it
        # plotted it and it looks good...

def test_xyProj():
    rayOrigin = [0,0,1]
    r = 0.5
    thetas = numpy.linspace(-numpy.pi, numpy.pi) # put in arctan2 domain
    x = r*numpy.cos(thetas)
    y = r*numpy.sin(thetas)
    z = [0.5]*len(x)
    px, py, pz, ps = proj2XYplane(x,y,z, rayOrigin)
    mags = numpy.sqrt(px**2+py**2)
    assert numpy.max(numpy.abs(mags-1)) < SMALL_NUM
    assert numpy.max(numpy.abs(pz)) < SMALL_NUM

    _thetas = numpy.arctan2(py, px)
    assert numpy.max(numpy.abs(_thetas-thetas)) < SMALL_NUM

    r = 1.5
    x = r*numpy.cos(thetas)
    y = r*numpy.sin(thetas)
    z = [-0.5]*len(x)
    _px, _py, _pz, _ps = proj2XYplane(x,y,z, rayOrigin)

    assert numpy.max(numpy.abs(px-_px)) < SMALL_NUM
    assert numpy.max(numpy.abs(py-_py)) < SMALL_NUM
    assert numpy.max(numpy.abs(_pz)) < SMALL_NUM

    _thetas = numpy.arctan2(_py, _px)
    assert numpy.max(numpy.abs(_thetas-thetas)) < SMALL_NUM

    rayOrigin = [0.5, 0.5, 100]
    x = 0.5
    y = 0.5
    z = 1
    px, py, pz, ps = proj2XYplane(x,y,z, rayOrigin)
    assert numpy.abs(px-x) < SMALL_NUM
    assert numpy.abs(py-y) < SMALL_NUM
    assert numpy.abs(pz) < SMALL_NUM

    rayOrigin = [-1, 0, 2]
    x = 0
    y = 0
    z = 1

    px, py, pz, ps = proj2XYplane(x,y,z, rayOrigin)
    assert numpy.abs(px-1) < SMALL_NUM
    assert numpy.abs(py) < SMALL_NUM
    assert numpy.abs(pz) < SMALL_NUM


def test_tangentToGuide():

    x = PIXEL_SIZE/1000 # one pixel from center in mm
    y = PIXEL_SIZE/1000 # one pixel from center in mm
    rayOrigin = [x, y, 100] # straight down projection
    z = 5

    expectPix = CHIP_CENTER + 1
    xPix, yPix, focusOff, isOK = tangentToGuide(x,y,z,rayOrigin)
    assert numpy.abs(xPix - expectPix) < SMALL_NUM
    assert numpy.abs(yPix - expectPix) < SMALL_NUM
    assert isOK
    assert numpy.abs(focusOff - z) < SMALL_NUM

    output = tangentToGuide(
        [x,x],
        [y,y],
        [z,z],
        rayOrigin
    )

    for _xPix, _yPix, _focusOff, _isOK in zip(*output):
        assert numpy.abs(xPix - expectPix) < SMALL_NUM
        assert numpy.abs(yPix - expectPix) < SMALL_NUM
        assert isOK
        assert numpy.abs(focusOff - z) < SMALL_NUM

    xPix, yPix, focusOff, isOK = tangentToGuide(
        [x,x*10000],  # out of range
        [y,y],
        [z,z],
        rayOrigin
    )

    assert isOK[0]
    assert not isOK[1]

    x = -1*PIXEL_SIZE/1000 # one pixel from center in mm
    y = -1*PIXEL_SIZE/1000 # one pixel from center in mm
    rayOrigin = [x, y, 100] # straight down projection
    z = 5

    expectPix = CHIP_CENTER - 1
    xPix, yPix, focusOff, isOK = tangentToGuide(x,y,z,rayOrigin)
    assert numpy.abs(xPix - expectPix) < SMALL_NUM
    assert numpy.abs(yPix - expectPix) < SMALL_NUM
    assert isOK
    assert numpy.abs(focusOff - z) < SMALL_NUM

    rayOrigin = [0, 0, 100]
    _xPix, _yPix, _focusOff, _isOK = tangentToGuide(x,y,z,rayOrigin)
    # slightly oblique projection
    assert xPix > _xPix
    assert yPix > _yPix
    assert _focusOff > focusOff
    assert _isOK

    xPix, yPix, focusOff, isOK = tangentToGuide(
        xTangent = 0,
        yTangent = 0,
        xBin = 2,
        yBin = 2
        )

    assert (xPix - CHIP_CENTER/2.) < SMALL_NUM
    assert (yPix - CHIP_CENTER/2.) < SMALL_NUM


def test_tangentGuideCycle():
    # in range pixels
    xPix = numpy.random.uniform(0,2*CHIP_CENTER, size=300)
    yPix = numpy.random.uniform(0,2*CHIP_CENTER, size=300)
    tx, ty = guideToTangent(xPix, yPix)
    _xPix, _yPix, _focusOff, _isOK = tangentToGuide(tx, ty)

    assert numpy.max(numpy.abs(xPix-_xPix)) < SMALL_NUM
    assert numpy.max(numpy.abs(yPix-_yPix)) < SMALL_NUM
    assert numpy.max(numpy.abs(_focusOff)) < SMALL_NUM

    # out of range pixels
    xPix = numpy.random.uniform(-2*CHIP_CENTER, 0, size=300)
    yPix = numpy.random.uniform(-2*CHIP_CENTER, 0, size=300)
    tx, ty = guideToTangent(xPix, yPix)
    _xPix, _yPix, _focusOff, _isOK = tangentToGuide(tx, ty)

    assert numpy.max(numpy.abs(xPix-_xPix)) < SMALL_NUM
    assert numpy.max(numpy.abs(yPix-_yPix)) < SMALL_NUM
    assert numpy.max(numpy.abs(_focusOff)) < SMALL_NUM
    assert True not in _isOK

    # x in range, y out of range pixels
    xPix = numpy.random.uniform(0, 2*CHIP_CENTER, size=300)
    yPix = numpy.random.uniform(-2*CHIP_CENTER, 0, size=300)
    tx, ty = guideToTangent(xPix, yPix)
    _xPix, _yPix, _focusOff, _isOK = tangentToGuide(tx, ty)
    assert numpy.max(numpy.abs(xPix-_xPix)) < SMALL_NUM
    assert numpy.max(numpy.abs(yPix-_yPix)) < SMALL_NUM
    assert numpy.max(numpy.abs(_focusOff)) < SMALL_NUM
    assert True not in _isOK

    # other side of range
    xPix = numpy.random.uniform(0, 2*CHIP_CENTER, size=300)
    yPix = numpy.random.uniform(2*CHIP_CENTER, 4*CHIP_CENTER, size=300)
    tx, ty = guideToTangent(xPix, yPix)
    _xPix, _yPix, _focusOff, _isOK = tangentToGuide(tx, ty)
    assert numpy.max(numpy.abs(xPix-_xPix)) < SMALL_NUM
    assert numpy.max(numpy.abs(yPix-_yPix)) < SMALL_NUM
    assert numpy.max(numpy.abs(_focusOff)) < SMALL_NUM
    assert True not in _isOK

    # look at projections
    # vary everything make sure round trip works
    for seed in range(1000):
        binX = numpy.random.choice([1,2,3])
        binY = numpy.random.choice([1,2,3])
        focusSide = numpy.random.choice([1, -1])
        xTangent = numpy.random.uniform(-5, 5)
        yTangent = numpy.random.uniform(-5, 5)
        zTangent = focusSide*numpy.random.uniform(1, 2)
        ox = numpy.random.uniform(-1, 1)
        oy = numpy.random.uniform(-1, 1)
        oz = numpy.random.uniform(90,100)
        rayOrigin = [ox,oy,oz]
        xPix, yPix, projDist, isOK = tangentToGuide(
            xTangent,
            yTangent,
            zTangent,
            rayOrigin,
            binX,
            binY
        )
        assert focusSide*projDist > 1
        assert focusSide*projDist < 2.5
        assert isOK
        _xt, _yt = guideToTangent(xPix, yPix, binX, binY)

        dist = numpy.sqrt((xTangent-_xt)**2+(yTangent-_yt)**2 + zTangent**2)
        assert numpy.abs(dist-focusSide*projDist) < SMALL_NUM

        dir1 = numpy.array([xTangent-ox, yTangent-oy, zTangent-oz])
        dir1 = dir1 / numpy.linalg.norm(dir1)

        dir2 = numpy.array([_xt-ox, _yt-oy, 0-oz])
        dir2 = dir2 / numpy.linalg.norm(dir2)

        # ensure they are the same vector!
        assert numpy.max(numpy.abs(dir1-dir2)) < SMALL_NUM


def test_tangentToPositioner():
    angErr = 0.0001 # degrees
    xBeta = 15
    yBeta = 0

    xt = 15 + 7.4
    yt = 0
    aExpect = 0
    bExpect = 0

    a, b, proj, isOK = tangentToPositioner(xt, yt, xBeta, yBeta)
    _xt, _yt = positionerToTangent(a,b,xBeta,yBeta)

    dist = numpy.sqrt((xt-_xt)**2+(yt-_yt)**2)*1000 # microns
    assert dist < 0.001
    assert numpy.abs(a-aExpect) < angErr
    assert numpy.abs(b-bExpect) < angErr
    assert isOK

    xt = 0
    yt = 15 + 7.4
    aExpect = 90
    bExpect = 0

    a, b, proj, isOK = tangentToPositioner(xt, yt, xBeta, yBeta)
    _xt, _yt = positionerToTangent(a,b,xBeta,yBeta)
    dist = numpy.sqrt((xt-_xt)**2+(yt-_yt)**2)*1000 # microns
    assert dist < 0.001
    assert numpy.abs(a-aExpect) < angErr
    assert numpy.abs(b-bExpect) < angErr
    assert isOK

    xt = -1*(15 + 7.4)
    yt = 0
    aExpect = 180
    bExpect = 0

    a, b, proj, isOK = tangentToPositioner(xt, yt, xBeta, yBeta)
    _xt, _yt = positionerToTangent(a,b,xBeta,yBeta)
    dist = numpy.sqrt((xt-_xt)**2+(yt-_yt)**2)*1000 # microns
    assert dist < 0.001
    assert numpy.abs(a-aExpect) < angErr
    assert numpy.abs(b-bExpect) < angErr
    assert isOK

    yt = -1*(15 + 7.4)
    xt = 0
    aExpect = 270
    bExpect = 0

    a, b, proj, isOK = tangentToPositioner(xt, yt, xBeta, yBeta)
    _xt, _yt = positionerToTangent(a,b,xBeta,yBeta)
    dist = numpy.sqrt((xt-_xt)**2+(yt-_yt)**2)*1000 # microns
    assert dist < 0.001
    assert numpy.abs(a-aExpect) < angErr
    assert numpy.abs(b-bExpect) < angErr
    assert isOK

    xt = (7.4 - 15)
    yt = 0
    aExpect = 0
    bExpect = 180

    a, b, proj, isOK = tangentToPositioner(xt, yt, xBeta, yBeta)
    _xt, _yt = positionerToTangent(a,b,xBeta,yBeta)
    dist = numpy.sqrt((xt-_xt)**2+(yt-_yt)**2)*1000 # microns
    assert dist < 0.001
    assert numpy.abs(a-aExpect) < angErr
    assert numpy.abs(b-bExpect) < angErr
    assert isOK

    xt = 0
    yt = 7.4 - 15
    aExpect = 90
    bExpect = 180

    a, b, proj, isOK = tangentToPositioner(xt, yt, xBeta, yBeta)
    _xt, _yt = positionerToTangent(a,b,xBeta,yBeta)
    dist = numpy.sqrt((xt-_xt)**2+(yt-_yt)**2)*1000 # microns
    assert dist < 0.001
    assert numpy.abs(a-aExpect) < angErr
    assert numpy.abs(b-bExpect) < angErr
    assert isOK

    xt = -1*(7.4 - 15)
    yt = 0
    aExpect = 180
    bExpect = 180

    a, b, proj, isOK = tangentToPositioner(xt, yt, xBeta, yBeta)
    _xt, _yt = positionerToTangent(a,b,xBeta,yBeta)
    dist = numpy.sqrt((xt-_xt)**2+(yt-_yt)**2)*1000 # microns
    assert dist < 0.001
    assert numpy.abs(a-aExpect) < angErr
    assert numpy.abs(b-bExpect) < angErr
    assert isOK

    xt = 0
    yt =  -1*(7.4 - 15)
    aExpect = 270
    bExpect = 180

    a, b, proj, isOK = tangentToPositioner(xt, yt, xBeta, yBeta)
    _xt, _yt = positionerToTangent(a,b,xBeta,yBeta)
    dist = numpy.sqrt((xt-_xt)**2+(yt-_yt)**2)*1000 # microns
    assert dist < 0.001
    assert numpy.abs(a-aExpect) < angErr
    assert numpy.abs(b-bExpect) < angErr
    assert isOK

def test_tangentToPositionerArrayInput():
    angErr = 0.0001 # degrees
    xBeta = 15
    yBeta = 0

    xts = [15 + 7.4, 0]
    yts = [0, 15 + 7.4]
    aExpects = [0, 90]
    bExpects = [0, 0]

    az, bs, projs, isOKs = tangentToPositioner(xts, yts, xBeta, yBeta)
    _xts, _yts = positionerToTangent(az,bs,xBeta,yBeta)

    zipMes = zip(xts, _xts, yts, _yts, az, bs, aExpects, bExpects, isOKs)
    for xt, _xt, yt, _yt, a, b, aExpect, bExpect, isOK in zipMes:
        dist = numpy.sqrt((xt-_xt)**2+(yt-_yt)**2)*1000 # microns
        assert dist < 0.001
        assert numpy.abs(a-aExpect) < angErr
        assert numpy.abs(b-bExpect) < angErr
        assert isOK

def test_tangetnPositionerOutOfRange():
    xBeta = 15
    yBeta = 0
    maxr = 15+7.4
    minr = 15-7.4

    rts = numpy.random.uniform(maxr, 4*maxr, size=1000)
    thetas = numpy.random.uniform(0, numpy.pi*2, size=1000)
    xts = rts*numpy.cos(thetas)
    yts = rts*numpy.sin(thetas)

    a, b, proj, isOK = tangentToPositioner(xts, yts, xBeta, yBeta)
    assert numpy.sum(numpy.isfinite(a)) == 0
    assert numpy.sum(numpy.isfinite(b)) == 0
    assert numpy.sum(isOK) == 0

    rts = numpy.random.uniform(0, minr, size=1000)
    thetas = numpy.random.uniform(0, numpy.pi*2, size=1000)
    xts = rts*numpy.cos(thetas)
    yts = rts*numpy.sin(thetas)

    a, b, proj, isOK = tangentToPositioner(xts, yts, xBeta, yBeta)
    assert numpy.sum(numpy.isfinite(a)) == 0
    assert numpy.sum(numpy.isfinite(b)) == 0
    assert numpy.sum(isOK) == 0

def test_tangentToPositionerOffCenter():
    # careful here, there is a degeneracy
    # near the edges of beta arm travel
    # where left hand and right hand
    # solutions are possible... but they
    # should never be commanded

    angErr = 0.0001 # degrees
    rBeta = 15
    offAng = numpy.radians(2) # 2 deg

    # this will fail:
    # xBeta = rBeta*numpy.cos(-offAng)
    # yBeta = rBeta*numpy.sin(-offAng)

    xBeta = rBeta*numpy.cos(offAng)
    yBeta = rBeta*numpy.sin(offAng)
    xt = 7.4 + xBeta
    yt = yBeta
    aExpect = 0
    bExpect = 0

    a, b, proj, isOK = tangentToPositioner(xt, yt, xBeta, yBeta)
    _xt, _yt = positionerToTangent(a,b,xBeta,yBeta)
    dist = numpy.sqrt((xt-_xt)**2+(yt-_yt)**2)*1000 # microns
    assert dist < 0.001
    assert numpy.abs(a-aExpect) < angErr
    assert numpy.abs(b-bExpect) < angErr
    assert isOK


def test_positionerTangentCycle():
    angErr = 0.0001 # degrees
    maxBetaCoordAngle = 4
    # hammer ths one because i'm worried about
    # degenerate solutions, left vs right hand...
    # because we have off axis targets on the beta arm
    # on axis targets can't be degenerate i think.
    npts = 1000000
    alphas = numpy.random.uniform(0,360, size=npts)

    #!!!! if you do this you'll expose the degeneracy problem !!!!
    # and the test will fail
    # betas = numpy.random.uniform(0.99*maxBetaCoordAngle,180-0.99*maxBetaCoordAngle, size=npts)
    # betas = numpy.random.uniform(0,180, size=npts)

    betas = numpy.random.uniform(maxBetaCoordAngle,180-maxBetaCoordAngle, size=npts)

    # beta arm coords of fiber
    rBeta = numpy.random.uniform(14.9,15.2, size=npts) # fiber radius in beta arm coords
    thetaBeta = numpy.random.uniform( # fiber off axis angle
        numpy.radians(-1*maxBetaCoordAngle),
        numpy.radians(maxBetaCoordAngle),
        size=npts
    )
    xBeta = rBeta*numpy.cos(thetaBeta)
    yBeta = rBeta*numpy.sin(thetaBeta)

    xt, yt = positionerToTangent(alphas, betas, xBeta, yBeta)
    _alphas, _betas, proj, isOK = tangentToPositioner(xt, yt, xBeta, yBeta)
    _xt, _yt = positionerToTangent(_alphas, _betas, xBeta, yBeta)

    maxDist = numpy.max(numpy.abs(numpy.sqrt((xt-_xt)**2+(yt-_yt)**2)))*1000
    maxAlpha =  numpy.max(numpy.abs(alphas-_alphas))
    maxBeta =  numpy.max(numpy.abs(betas-_betas))
    print(maxAlpha, maxBeta)
    assert maxDist < 0.001
    assert maxAlpha < angErr
    assert maxBeta < angErr


def test_tangentPositionerCycle():
    angErr = 0.0001 # degrees
    maxBetaCoordAngle = numpy.radians(4)
    npts = 1000000
    # i think this direction can't suffer the
    # left/right hand degeneracy, this direction
    # always yeields the right hand config.
    maxBetaR = 15.1
    minBetaR = 14.9
    # beta arm coords
    betaRs = numpy.random.uniform(minBetaR, maxBetaR, size=npts)
    betaThetas = numpy.random.uniform(
        -1*maxBetaCoordAngle, maxBetaCoordAngle, size=npts
    )
    xBeta = betaRs*numpy.cos(betaThetas)
    yBeta = betaRs*numpy.sin(betaThetas)

    # tangent coords
    tr = numpy.random.uniform(maxBetaR-7.4 , minBetaR-2, size=npts)
    tTheta = numpy.random.uniform(0, numpy.pi*2, size=npts)
    xt = tr*numpy.cos(tTheta)
    yt = tr*numpy.sin(tTheta)
    alphas, betas, proj, isOK = tangentToPositioner(xt, yt, xBeta, yBeta)
    _xt, _yt = positionerToTangent(alphas, betas, xBeta, yBeta)
    _alphas, _betas, _proj, _isOK = tangentToPositioner(_xt, _yt, xBeta, yBeta)

    maxDist = numpy.max(numpy.abs(numpy.sqrt((xt-_xt)**2+(yt-_yt)**2)))*1000
    maxAlpha =  numpy.max(numpy.abs(alphas-_alphas))
    maxBeta =  numpy.max(numpy.abs(betas-_betas))
    assert maxDist < 0.001
    assert maxAlpha < angErr
    assert maxBeta < angErr
    assert not False in isOK
    assert not False in _isOK


def test_tangentPositionerProjection():
    angErr = 0.0001
    for seed in range(100):
        ox = numpy.random.uniform(-1,1)
        oy = numpy.random.uniform(-1,1)
        oz = numpy.random.uniform(900, 1100)
        rayOrigin = [ox, oy, oz]
        xBeta = numpy.random.uniform(14.9, 15.1)
        yBeta = numpy.random.uniform(-0.1, 0.1)
        rt = numpy.random.uniform(14, 18)
        rtheta = numpy.random.uniform(0, numpy.pi*2)
        xt = rt*numpy.cos(rtheta)
        yt = rt*numpy.sin(rtheta)
        zt = numpy.random.uniform(-4,4)
        alpha, beta, proj, isOK = tangentToPositioner(
            xt, yt, xBeta, yBeta, zTangent=zt, rayOrigin=rayOrigin
        )
        _xt, _yt = positionerToTangent(alpha, beta, xBeta, yBeta)

        _alpha, _beta, _proj, _isOK = tangentToPositioner(
            _xt, _yt, xBeta, yBeta
        )

        dist = numpy.sqrt((xt-_xt)**2+(yt-_yt)**2 + zt**2)
        assert numpy.abs(dist-numpy.sign(zt)*proj) < SMALL_NUM

        dir1 = numpy.array([xt-ox, yt-oy, zt-oz])
        dir1 = dir1 / numpy.linalg.norm(dir1)

        dir2 = numpy.array([_xt-ox, _yt-oy, 0-oz])
        dir2 = dir2 / numpy.linalg.norm(dir2)

        # ensure they are the same vector!
        assert numpy.max(numpy.abs(dir1-dir2)) < SMALL_NUM

        assert isOK
        assert _isOK
        assert numpy.abs(alpha-_alpha) < angErr
        assert numpy.abs(beta-_beta) < angErr



def test_tangentPositionerProjectionArray():
    # blatant cut and paste from previous test
    # addint the z tanget coord
    angErr = 0.0001 # degrees
    maxBetaCoordAngle = numpy.radians(4)
    npts = 1000000
    rayOrigin = [0,0,1000]
    # i think this direction can't suffer the
    # left/right hand degeneracy, this direction
    # always yeields the right hand config.
    maxBetaR = 15.1
    minBetaR = 14.9
    # beta arm coords
    # betaRs = numpy.random.uniform(minBetaR, maxBetaR, size=npts)
    # betaThetas = numpy.random.uniform(
    #     -1*maxBetaCoordAngle, maxBetaCoordAngle, size=npts
    # )
    xBeta = numpy.random.uniform(14.9, 15.1, size=npts)
    yBeta = numpy.random.uniform(-0.1, 0.1, size=npts)

    # xBeta = betaRs*numpy.cos(betaThetas)
    # yBeta = betaRs*numpy.sin(betaThetas)

    # tangent coords
    tr = numpy.random.uniform(10 , 20, size=npts)
    tTheta = numpy.random.uniform(0, numpy.pi*2, size=npts)
    xt = tr*numpy.cos(tTheta)
    yt = tr*numpy.sin(tTheta)
    zt = numpy.random.uniform(-1,1, size=npts)


    alphas, betas, proj, isOK = tangentToPositioner(
        xt, yt, xBeta, yBeta, zTangent=zt, rayOrigin=rayOrigin
    )
    # print("out alpha", alphas, betas)
    _xt, _yt = positionerToTangent(alphas, betas, xBeta[0], yBeta[0])

    # don't project on this one! its been projected
    _alphas, _betas, _proj, _isOK = tangentToPositioner(
        _xt, _yt, xBeta[0], yBeta[0]
    )
    # print("out alpha2", _alphas, _betas)
    # print("dists mm, ", numpy.sqrt((xt-_xt)**2+(yt-_yt)**2))
    # maxDist = numpy.max(numpy.sqrt((xt-_xt)**2+(yt-_yt)**2))*1000
    maxAlpha = numpy.max(numpy.abs(alphas-_alphas))
    maxBeta = numpy.max(numpy.abs(betas-_betas))

    # assert maxDist < 0.001
    assert maxAlpha < angErr
    assert maxBeta < angErr
    assert not False in isOK
    assert not False in _isOK


if __name__ == "__main__":
    test_wokTangentCycle()

