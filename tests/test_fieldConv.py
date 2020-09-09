import pytest
import numpy
import matplotlib.pyplot as plt
from sdssconv.fieldCoords import cart2FieldAngle, fieldAngle2Cart
from sdssconv.fieldCoords import sph2Cart, cart2Sph, SMALL_NUM
from sdssconv.fieldCoords import observedToField, fieldToObserved
from sdssconv.fieldCoords import focalToField, fieldToFocal, focalPlaneModelDict

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
            g = x*_x + y*_y + z*_z
            # numerical overflow makes g > 1 sometimes?
            # this screws up the arccosine below, so round it
            g = numpy.round(g,10)
            # print("min max", numpy.min(g), numpy.max(g))
            angSep = numpy.degrees(numpy.arccos(g))*3600
            assert numpy.max(angSep) < 0.0001 # arcsec, basically no error



if __name__ == "__main__":
    test_fieldFocalCycle()
