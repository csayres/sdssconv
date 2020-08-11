import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)


def parseEdraw(edrawPath):
    """Load xyz positions from wok surface model measured (painfully) with
    edrawings clicks

    mapping from edrawing coord sys to wok coord sys
    X -> X
    Y -> Z
    Z -> -Y
    """
    with open(edrawPath) as f:
        lines = f.readlines()

    posXYZ = [] # only top half of positioners are measured
    dias = []
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            # a comment
            continue
        line = line.strip("mm")
        if line.startswith("Diameter: "):
            dia = float(line.strip("Diameter: "))
            dia = numpy.around(dia, 2)
        if line.startswith("X:"):
            x = float(line.strip("X: "))
        elif line.startswith("Y: "):
            z = float(line.strip("Y: "))
        elif line.startswith("Z: "):
            y = -1*float(line.strip("Z: "))
            posXYZ.append([x, y, z])
            dias.append(dia)
    dias = numpy.array(dias)

    # reflect the top half (+y) onto lower half (-y)
    # (I only measured half of the array because its a terrible job
    # and it must be symmetric about the x axis).
    topXYZ = numpy.array(posXYZ)
    bottomXYZ = numpy.copy(topXYZ)
    # don't duplicate the y=0 row (equator)
    inds = numpy.argwhere(topXYZ[:,1] > 0)
    bottomXYZ = bottomXYZ[inds].squeeze()
    bottomXYZ[:,1] = -1*bottomXYZ[:,1]
    xyz = numpy.vstack((topXYZ, bottomXYZ))
    dias = numpy.hstack((dias, dias[inds].squeeze()))
    r = numpy.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
    df = pd.DataFrame(
        {
            "x": xyz[:,0],
            "y": xyz[:,1],
            "z": xyz[:,2],
            "r": r,
            "holeDia": numpy.array(dias)
        }
    )
    return df

def parseFilledHex():
    """
    """
    gridFile = os.path.join(os.environ["SDSSCONV_DIR"], "data/fps_filledHex.txt")
    # Row   Col     X (mm)  Y (mm)          Assignment
    #
    #-13   0 -145.6000 -252.1866  BA
    #-13   1 -123.2000 -252.1866  BA
    # bossXY = []
    # baXY = []
    # fiducialXY = []
    xPos = []
    yPos = []
    holeType = []
    row = []
    col = []
    with open(gridFile, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split("#")[0]
        if not line:
            continue
        row, col, x, y, fType = line.split()
        x = float(x)
        y = float(y)
        xPos.append(x)
        yPos.append(y)
        holeType.append(fType)
    xPos = numpy.array(xPos)
    yPos = numpy.array(yPos)
    r = numpy.sqrt(xPos**2 + yPos**2)
    df = pd.DataFrame(
        {
            "x": xPos,
            "y": yPos,
            "r": r,
            "holeType": holeType,
            "row": row,
            "col": col
        }
    )
    return df


def compileDataIGES():
    """Compile information from Ricks filled hex coords, and the
    coords extracted from the IGES data sent from tapemation

    must run $IGES_DIR/read_IGES.py to produce the required csvs
    these csvs (and parent IGS files) are copied to the data directory

    creates a pandas dataframe and saves as csv data/wokCoords.csv

    returns
    -------
    result : pd.DataFrame
        dataframe containing coordinates and directions for all holes in a wok, including type
        assignments.
    """
    lcoDF = pd.read_csv(os.path.join(os.getenv("SDSSCONV_DIR"), "data/duPontWokHoles.csv"))
    apoDF = pd.read_csv(os.path.join(os.getenv("SDSSCONV_DIR"), "data/sloanWokHoles.csv"))

    # rough positions of outer gfa/fiducial ring
    # measurements from edrawings + wok solid models
    outerFidLCOradius = 329.37
    outerFidAPOradius = 329.04
    outerFidTheta = [
        0, 15, 45, 60, 75, 105, 120, 135, 165, 180, 195, 225,
        240, 255, 285, 300, 315, 345
    ]
    gfaLCOradius = 333.42
    gfaLCOPinRadius = 379.386
    gfaAPOradius = 333.09
    gfaAPOPinRadius = 379.065
    gfaTheta = [30, 90, 150, 210, 270, 330]

    mountHoleSep = 25/2. # distance from hole center to mounting hole

    wokType = []
    holeType = []
    holeID = []
    # coords from rick's filled hex pattern
    # xFP = []
    # yFP = []
    # coords for center point of hole on wok, base position of robot
    xWok = []
    yWok = []
    zWok = []
    rWok = []
    thetaWok = []
    # k direction unit vector along axis of hole (roughly -z to +z), defines
    # a positioner/gfa z axis
    kx = []
    ky = []
    kz = []
    # i direction unit vector pointing from positioner base to -y mounting hole
    # defines a positioners x axis, for gfa i axis found by cross product j x k
    ix = []
    iy = []
    iz = []
    # the final vector (positioners y axis)
    # is found via cross products k x i
    # for GFA j is direction pointing toward pin
    jx = []
    jy = []
    jz = []

    _distChecks = []
    _normChecks = []
    def findHole(xTest, yTest, xPinOff, yPinOff, df, _wokType, _holeType, _holeID):
        xyzTop = df[["x1", "y1", "z1"]].to_numpy() # wok top
        xyzBottom = df[["x2", "y2", "z2"]].to_numpy() # wok bottom
        amin = numpy.argmin((xyzTop[:,0]-xTest)**2 + (xyzTop[:,1]-yTest)**2)

        xyz = xyzTop[amin] # wok xyz position

        dist1 = numpy.linalg.norm([xyz[0]-xTest, xyz[1]-yTest])
        _distChecks.append(dist1)
        # construct unit vector k along hole axis
        k = xyzBottom[amin]
        k = xyz - k
        k = k / numpy.linalg.norm(k) # unit vector

        # find the mount holes
        amin = numpy.argmin((xyzTop[:,0] - xyz[0] + xPinOff)**2 + (xyzTop[:,1] - xyz[1] + yPinOff)**2)

        # orthogonally project this position on to the k direction

        m = xyzTop[amin] # mount hole or GFA pin
        # print("m-b", m-xyz)
        dist2 = numpy.linalg.norm([m[0]-xyz[0]+xPinOff, m[1]-xyz[1]+yPinOff])
        _distChecks.append(dist2)
        _k = k.reshape((3,1))
        if _holeType == "GFA":
            # orthogonal projection from pin is the j axis (+y)
            j = (m-xyz) - _k.dot(_k.T).dot(m-xyz)
            j = j / numpy.linalg.norm(j)

            # find 3rd axis by cross product
            i = numpy.cross(j, k)
        else:
            # orthogonal projection from hole is the i axis (+x)
            i = (m-xyz) - _k.dot(_k.T).dot(m-xyz)
            i = i / numpy.linalg.norm(i)

            # find 3rd axis by cross product
            j = numpy.cross(k, i)

        _normChecks.append(i.dot(j))
        _normChecks.append(i.dot(k))

        # construct unit vector through mounting holes
        # tangent to wok surface
        # lmn = xyzTop[aminTop,:] - xyzTop[aminBottom,:]
        # lmn = lmn / numpy.linalg.norm(lmn) # unit vector

        # they should be perpendicular
        # numerically they are to ~3 decimal points in degrees
        # could use orthogonal projections to improve it
        # d = numpy.degrees(numpy.arccos(ijk.dot(lmn)))
        # print("d", d, dist1*1000, dist2*1000)

        wokType.append(_wokType)
        holeType.append(_holeType)
        holeID.append(_holeID)
        # xFP.append(numpy.nan)
        # yFP.append(numpy.nan)
        xWok.append(xyz[0])
        yWok.append(xyz[1])
        zWok.append(xyz[2])
        rWok.append(numpy.sqrt(xyz[0]**2+xyz[1]**2))
        _thetaWok = numpy.degrees(numpy.arctan2(xyz[1], xyz[0]))
        if _thetaWok < 0:
            _thetaWok += 360
        thetaWok.append(_thetaWok)
        ix.append(i[0])
        iy.append(i[1])
        iz.append(i[2])
        jx.append(j[0])
        jy.append(j[1])
        jz.append(j[2])
        kx.append(k[0])
        ky.append(k[1])
        kz.append(k[2])

    holeCounter = 0
    # ricks filled hex search
    fh = parseFilledHex()
    for index, row in fh.iterrows():
        xTest = row["x"]
        yTest = row["y"]
        _ht = row["holeType"]
        # rename hole types from ricks file
        # to be slightly more descriptive
        if _ht == "BA":
            ht = "ApogeeBoss"
        elif _ht == "BOSS":
            ht = "Boss"
        else:
            # fiducial don't rename it
            ht = _ht
        for wt, df in zip(["LCO", "APO"], [lcoDF, apoDF]):
            findHole(xTest, yTest, 0, mountHoleSep, df, wt, ht, holeCounter)
        holeCounter += 1

    # outer fiducial ring search
    for theta in outerFidTheta:
        for wt, df, radius in zip(["LCO", "APO"], [lcoDF, apoDF], [outerFidLCOradius, outerFidAPOradius]):
            xTest = radius*numpy.cos(numpy.radians(theta))
            yTest = radius*numpy.sin(numpy.radians(theta))
            findHole(xTest, yTest, 0, mountHoleSep, df, wt, "Fiducial", holeCounter)
        holeCounter += 1

    # GFA hole search
    for theta in gfaTheta:
        for wt, df, radius, pinRadius in zip(["LCO", "APO"], [lcoDF, apoDF], [gfaLCOradius, gfaAPOradius], [gfaLCOPinRadius, gfaAPOPinRadius]):
            xTest = radius*numpy.cos(numpy.radians(theta))
            yTest = radius*numpy.sin(numpy.radians(theta))
            xPinOff = xTest - pinRadius*numpy.cos(numpy.radians(theta))
            yPinOff = yTest - pinRadius*numpy.sin(numpy.radians(theta))
            findHole(xTest, yTest, xPinOff, yPinOff, df, wt, "GFA", holeCounter)
        holeCounter += 1

    # print("max dist", numpy.max(_distChecks))
    # print("max norm", numpy.max(_normChecks))

    df = pd.DataFrame(
        {
            "wokType": wokType,
            "holeType": holeType,
            "holeID": holeID,
            "x": xWok,
            "y": yWok,
            "z": zWok,
            "r": rWok,
            "theta": thetaWok,
            "ix": ix,
            "iy": iy,
            "iz": iz,
            "jx": jx,
            "jy": jy,
            "jz": jz,
            "kx": kx,
            "ky": ky,
            "kz": kz,
        }
    )

    df.to_csv(os.path.join(os.getenv("SDSSCONV_DIR"), "data/wokCoords.csv"))
    # populate inner hex
    # _df = df[df["holeType"] != "GFA"]
    # _df = _df[["ix", "iy", "iz"]]
    # print(_df)

    # print("total", len(wokType))

    # import pdb; pdb.set_trace()

    # rough positions of inner hex pattern


def compileDataEdraw():
    """Compile information from Rick's filled hex coords, and the coords
    extracted from the solid model hole positions.

    returns
    -------
    result : pd.DataFrame
        dataframe containing coordinates and directions for all holes in a wok, including type
        assignments.
    """
    fh = parseFilledHex()
    topLCO = parseEdraw(os.path.join(os.getenv("SDSSCONV_DIR"), "data/LCOEdrawExtract.txt"))
    tlcoxy = topLCO[["x", "y", "z", "holeDia"]].to_numpy()
    bottomLCO = parseEdraw(os.path.join(os.getenv("SDSSCONV_DIR"), "data/LCOEdrawExtractBackside.txt"))
    blcoxy = bottomLCO[["x", "y", "z", "holeDia"]].to_numpy()
    topAPO = parseEdraw(os.path.join(os.getenv("SDSSCONV_DIR"), "data/APOEdrawExtract.txt"))
    tapoxy = topAPO[["x", "y", "z", "holeDia"]].to_numpy()
    bottomAPO = parseEdraw(os.path.join(os.getenv("SDSSCONV_DIR"), "data/APOEdrawExtractBackside.txt"))
    bapoxy = bottomAPO[["x", "y", "z", "holeDia"]].to_numpy()

    holeDias = []
    holeTypes = []
    wokTypes = []
    # xy coords in focal surface
    xFP = []
    yFP = []
    # xyz position on wok surface
    xWok = []
    yWok = []
    zWok = []
    # ijk direction normal to wok surface at point xyz
    iWok = []
    jWok = []
    kWok = []
    # first the positioner field
    for index, row in fh.iterrows():
        x = row["x"]
        y = row["y"]
        _holeType = row["holeType"]
        if _holeType == "BA":
            holeType = "ApogeeBoss"
        elif _holeType == "BOSS":
            holeType = "Boss"
        else:
            # fiducial don't rename it
            holeType = _holeType

        aminapo = numpy.argmin((tapoxy[:,0]-x)**2 + (tapoxy[:,1]-y)**2)
        _aminapo = numpy.argmin((bapoxy[:,0]-x)**2 + (bapoxy[:,1]-y)**2)

        aminlco = numpy.argmin((tlcoxy[:,0]-x)**2 + (tlcoxy[:,1]-y)**2)
        _aminlco = numpy.argmin((blcoxy[:,0]-x)**2 + (blcoxy[:,1]-y)**2)

        # apo wok stuff
        apoWokxyz = tapoxy[aminapo,:3]
        apoWokijk = apoWokxyz - bapoxy[_aminapo,:3]
        apoWokijk = apoWokijk / numpy.linalg.norm(apoWokijk)
        xFP.append(x)
        yFP.append(y)
        xWok.append(apoWokxyz[0])
        yWok.append(apoWokxyz[1])
        zWok.append(apoWokxyz[2])
        iWok.append(apoWokijk[0])
        jWok.append(apoWokijk[1])
        kWok.append(apoWokijk[2])
        holeDias.append(tapoxy[aminapo,-1])
        holeTypes.append(holeType)
        wokTypes.append("APO")


        # lco wok stuff
        lcoWokxyz = tlcoxy[aminlco,:3]
        lcoWokijk = lcoWokxyz - blcoxy[_aminlco,:3]
        lcoWokijk = lcoWokijk / numpy.linalg.norm(lcoWokijk)
        xFP.append(x)
        yFP.append(y)
        xWok.append(lcoWokxyz[0])
        yWok.append(lcoWokxyz[1])
        zWok.append(lcoWokxyz[2])
        iWok.append(lcoWokijk[0])
        jWok.append(lcoWokijk[1])
        kWok.append(lcoWokijk[2])
        holeDias.append(tlcoxy[aminlco,-1])
        holeTypes.append(holeType)
        wokTypes.append("LCO")

    # handle central hole (its empty but drilled)
    aminapo = numpy.argmin((tapoxy[:,0])**2 + (tapoxy[:,1])**2)
    _aminapo = numpy.argmin((bapoxy[:,0])**2 + (bapoxy[:,1])**2)
    apoWokxyz = tapoxy[aminapo,:3]
    apoWokijk = apoWokxyz - bapoxy[_aminapo,:3]
    apoWokijk = apoWokijk / numpy.linalg.norm(apoWokijk)
    xFP.append(0)
    yFP.append(0)
    xWok.append(apoWokxyz[0])
    yWok.append(apoWokxyz[1])
    zWok.append(apoWokxyz[2])
    iWok.append(apoWokijk[0])
    jWok.append(apoWokijk[1])
    kWok.append(apoWokijk[2])
    holeDias.append(tapoxy[aminapo,-1])
    holeTypes.append("Empty")
    wokTypes.append("APO")

    aminlco = numpy.argmin((tlcoxy[:,0])**2 + (tlcoxy[:,1])**2)
    _aminlco = numpy.argmin((blcoxy[:,0])**2 + (blcoxy[:,1])**2)
    lcoWokxyz = tlcoxy[aminlco,:3]
    lcoWokijk = lcoWokxyz - blcoxy[_aminlco,:3]
    lcoWokijk = lcoWokijk / numpy.linalg.norm(lcoWokijk)
    xFP.append(0)
    yFP.append(0)
    xWok.append(lcoWokxyz[0])
    yWok.append(lcoWokxyz[1])
    zWok.append(lcoWokxyz[2])
    iWok.append(lcoWokijk[0])
    jWok.append(lcoWokijk[1])
    kWok.append(lcoWokijk[2])
    holeDias.append(tlcoxy[aminlco,-1])
    holeTypes.append("Empty")
    wokTypes.append("LCO")

    # handle outer fiducial ring and gfa holes
    t = topLCO[topLCO["r"] > 310]
    for index, row in t.iterrows():
        xyz = row[["x", "y", "z"]].to_numpy()
        # find corresponding bottom hole
        amin = numpy.argmin((blcoxy[:,0]-xyz[0])**2 + (blcoxy[:,1]-xyz[1])**2)

        lcoWokijk = lcoWokxyz - blcoxy[amin,:3]
        lcoWokijk = lcoWokijk / numpy.linalg.norm(lcoWokijk)
        xFP.append(numpy.nan)
        yFP.append(numpy.nan)
        xWok.append(xyz[0])
        yWok.append(xyz[1])
        zWok.append(xyz[2])
        iWok.append(lcoWokijk[0])
        jWok.append(lcoWokijk[1])
        kWok.append(lcoWokijk[2])
        holeDia = tlcoxy[amin,-1]
        holeDias.append(holeDia)
        if holeDia > 30:
            holeTypes.append("GFA")
        else:
            holeTypes.append("Fiducial")
        wokTypes.append("LCO")


    t = topAPO[topAPO["r"] > 310]
    for index, row in t.iterrows():
        xyz = row[["x", "y", "z"]].to_numpy()
        # find corresponding bottom hole
        amin = numpy.argmin((bapoxy[:,0]-xyz[0])**2 + (bapoxy[:,1]-xyz[1])**2)

        apoWokijk = apoWokxyz - bapoxy[amin,:3]
        apoWokijk = apoWokijk / numpy.linalg.norm(apoWokijk)
        xFP.append(numpy.nan)
        yFP.append(numpy.nan)
        xWok.append(xyz[0])
        yWok.append(xyz[1])
        zWok.append(xyz[2])
        iWok.append(apoWokijk[0])
        jWok.append(apoWokijk[1])
        kWok.append(apoWokijk[2])
        holeDia = tapoxy[amin,-1]
        holeDias.append(holeDia)
        if holeDia > 30:
            holeTypes.append("GFA")
        else:
            holeTypes.append("Fiducial")
        wokTypes.append("APO")

    xWok = numpy.array(xWok)
    yWok = numpy.array(yWok)
    xFP = numpy.array(xFP)
    yFP = numpy.array(yFP)

    rWok = numpy.sqrt(xWok**2 + yWok**2)
    rFP = numpy.sqrt(xFP**2 + yFP**2)
    thetaWok = numpy.degrees(numpy.arctan2(yWok, xWok))
    ind = numpy.argwhere(thetaWok < 0).flatten()
    thetaWok[ind] = thetaWok[ind] + 360
    thetaFP = numpy.degrees(numpy.arctan2(yFP, xFP))
    ind = numpy.argwhere(thetaFP < 0).flatten()
    thetaFP[ind] = thetaFP[ind] + 360

    df = pd.DataFrame(
        {
            "wokType": wokTypes,
            "holeType": holeTypes,
            "holeDia": holeDias,
            "xFP": xFP,
            "yFP": yFP,
            "rFP": rFP,
            "thetaFP": thetaFP,
            "xWok": xWok,
            "yWok": yWok,
            "zWok": zWok,
            "rWok": rWok,
            "thetaWok": thetaWok,
            "iWok": iWok,
            "jWok": jWok,
            "kWok": kWok
        }
    )

    # determine the z position at 0,0
    # reference to there
    _df = df[df["wokType"] == "LCO"]
    amin = numpy.argmin(_df["rWok"].to_numpy())
    zLCO = _df["zWok"].to_numpy()[amin]

    _df = df[df["wokType"] == "APO"]
    amin = numpy.argmin(_df["rWok"].to_numpy())
    zAPO = _df["zWok"].to_numpy()[amin]

    zOffsets = []
    for wt in df["wokType"]:
        if wt == "APO":
            zOffsets.append(zAPO)
        else:
            zOffsets.append(zLCO)
    zOffsets = numpy.array(zOffsets)
    df["zWok"] = df["zWok"] - zOffsets

    return df


def wokCurveAPO(r):
    """The curve of the wok at APO at radial position r

    Parameters
    -----------
    r : scalar or 1D array
        radius (cylindrical coords) mm

    Returns:
    ---------
    result : scalar or 1D array
        z (height) of wok surface in mm (0 at vertex)
    """
    A = 9199.322517101522
    return A - numpy.sqrt(A**2 - r**2)


def wokSlopeAPO(r):
    """The slope of the wok at APO at radial position r

    Parameters
    -----------
    r : scalar or 1D array
        radius (cylindrical coords) mm

    Returns:
    ---------
    result : scalar or 1D array
        dz/dr (slope) of wok surface
    """
    A = 9199.322517101522
    return r/numpy.sqrt(A**2-r**2)


def wokCurveLCO(r):
    """The curve of the wok at LCO at radial position r

    Parameters
    -----------
    r : scalar or 1D array
        radius (cylindrical coords) mm

    Returns:
    ---------
    result : scalar or 1D array
        z (height) of wok surface in mm (0 at vertex)
    """
    A = 0.000113636363636
    B = 0.0000000129132231405
    C = 0.0000012336318
    return A*r**2/(1+numpy.sqrt(1-B*r**2)) + C*r**2

def wokSlopeLCO(r):
    """The slope of the wok at LCO at radial position r

    Parameters
    -----------
    r : scalar or 1D array
        radius (cylindrical coords) mm

    Returns:
    ---------
    result : scalar or 1D array
        dz/dr (slope) of wok surface
    """
    A = 0.000113636363636
    B = 0.0000000129132231405
    C = 0.0000012336318
    return A*r/numpy.sqrt(1-B*r**2) + 2*C*r

def plotDirectionsEdraw():
    """Plot the residuals between
    math derived tangent direction for positioners
    and model derived direction for positioners
    """
    df = compileDataEdraw()
    _df = df[pd.notna(df["xFP"])]
    lco = _df[_df["wokType"] == "LCO"]

    # print(lco[["xFP", "yFP"]])

    rs = []
    residuals = []
    phi1 = []
    phiSolid = []
    for index, row in lco.iterrows():
        rFP = numpy.sqrt(row["xFP"]**2+row["yFP"]**2)
        # rFP = numpy.sqrt(row["xWok"]**2+row["yWok"]**2)
        phiFP = 90 - numpy.degrees(numpy.arctan(wokSlopeLCO(rFP)))
        A = row[["xWok","yWok","zWok"]].to_numpy()
        B = row[["iWok","jWok","kWok"]].to_numpy()
        toCen = numpy.array([0-A[0], 0-A[1], 0])
        toCen = toCen / numpy.linalg.norm(toCen)
        phiModel = numpy.degrees(numpy.arccos(B.dot(toCen)))
        phi1.append(phiFP)
        phiSolid.append(phiModel)
        rs.append(rFP)

    phiSolid = numpy.array(phiSolid)
    phi1 = numpy.array(phi1)

    print("phi solid", phiSolid)
    plt.figure()
    # plt.plot(rs, phi1-phiSolid, label="resid")
    plt.plot(rs, 143*1000*numpy.sin(numpy.radians((phi1-phiSolid))), label="resid")
    plt.ylabel("tangential error at tip (um)")
    plt.xlabel("r (mm)")
    # plt.plot(rs, phiSolid, label="phiSolid")
    plt.legend()
    plt.title("LCO")


    apo = _df[_df["wokType"]=="APO"]

    # print(apo[["xFP", "yFP"]])

    rs = []
    residuals = []
    phi1 = []
    phiSolid = []
    for index, row in apo.iterrows():
        rFP = numpy.sqrt(row["xFP"]**2+row["yFP"]**2)
        # rFP = numpy.sqrt(row["xWok"]**2+row["yWok"]**2)
        phiFP = 90 - numpy.degrees(numpy.arctan(wokSlopeAPO(rFP)))
        A = row[["xWok","yWok","zWok"]].to_numpy()
        B = row[["iWok","jWok","kWok"]].to_numpy()
        toCen = numpy.array([0-A[0], 0-A[1], 0])
        toCen = toCen / numpy.linalg.norm(toCen)
        phiModel = numpy.degrees(numpy.arccos(B.dot(toCen)))
        phi1.append(phiFP)
        phiSolid.append(phiModel)
        rs.append(rFP)

    phiSolid = numpy.array(phiSolid)
    phi1 = numpy.array(phi1)

    print("phi solid", phiSolid)
    plt.figure()
    plt.plot(rs, 143*1000*numpy.sin(numpy.radians((phi1-phiSolid))), label="resid")
    plt.ylabel("tangential error at tip (um)")
    plt.xlabel("r (mm)")
    # plt.plot(rs, phiSolid, label="phiSolid")
    plt.legend()
    plt.title("APO")

    plt.show()


def plotHoleTypes():
    """Plot xyz and hole type information
    """
    df = compileDataEdraw()
    plt.figure(figsize=(10,10))
    sns.scatterplot(x="xWok", y="yWok", hue="zWok", style="holeType", data=df[df["wokType"]=="LCO"])
    ax = plt.gca()
    ax.axis("equal")
    plt.title("LCO")

    plt.figure(figsize=(10,10))
    sns.scatterplot(x="xWok", y="yWok", hue="zWok", style="holeType", data=df[df["wokType"]=="APO"])
    ax = plt.gca()
    ax.axis("equal")
    plt.title("APO")

    plt.figure(figsize=(5,5))
    sns.scatterplot(x="rWok", y="zWok", data=df[df["wokType"]=="APO"])
    plt.title("APO")

    plt.figure(figsize=(5,5))
    sns.scatterplot(x="rWok", y="zWok", data=df[df["wokType"]=="LCO"])
    plt.title("LCO")

    plt.show()


def plotIGES():
    # LCO color_number map
    # 7 = axis
    # 4, 5 = junk?
    # 8 = pins and tapped
    lcoDF = pd.read_csv(os.path.join(os.getenv("SDSSCONV_DIR"), "data/duPontWokHoles.csv"))
    plt.figure(figsize=(8,8))
    # lcoDF = lcoDF[lcoDF["color_number"]==5]
    sns.scatterplot(x="x2", y="y2", hue="z2", style="color_number", data=lcoDF)
    ax = plt.gca()
    ax.axis("equal")
    plt.title("LCO 2")

    plt.figure(figsize=(8,8))
    # lcoDF = lcoDF[lcoDF["color_number"]==5]
    sns.scatterplot(x="x1", y="y1", hue="z1", style="color_number", data=lcoDF)
    ax = plt.gca()
    ax.axis("equal")
    plt.title("LCO 1")


    # APO color_number map
    # 7 = axis
    # 2, 4 = junk?
    # 8 = pins and tapped
    apoDF = pd.read_csv(os.path.join(os.getenv("SDSSCONV_DIR"), "data/sloanWokHoles.csv"))

    plt.figure(figsize=(8,8))
    # lcoDF = lcoDF[lcoDF["color_number"]==5]
    sns.scatterplot(x="x2", y="y2", hue="z2", style="color_number", data=apoDF)
    ax = plt.gca()
    plt.title("APO 2")
    ax.axis("equal")

    plt.figure(figsize=(8,8))
    # apoDF = apoDF[apoDF["color_number"]==5]
    sns.scatterplot(x="x1", y="y1", hue="z1", style="color_number", data=apoDF)
    ax = plt.gca()
    plt.title("APO 1")
    ax.axis("equal")


    plt.figure(figsize=(8,8))
    sns.scatterplot(x="r2", y="z2", style="color_number", data=lcoDF)
    plt.title("LCO")

    plt.figure(figsize=(8,8))
    sns.scatterplot(x="r2", y="z2", style="color_number", data=apoDF)
    plt.title("APO")

    plt.show()

    # plt.plot(lcoDF["r1"], lcoDF["z1"], '.', label="LCO 1")
    # plt.plot(lcoDF["r2"], lcoDF["z2"], '.', label="LCO 2")
    # plt.plot(apoDF["r1"], apoDF["z1"], '.', label="APO 1")
    # plt.plot(apoDF["r2"], apoDF["z2"], '.', label="APO 2")

    # plt.legend()

    # plt.show()

def plotCompiledData():
    df = pd.read_csv(os.path.join(os.getenv("SDSSCONV_DIR"), "data/wokCoords.csv"))
    lco = df[df["wokType"]=="LCO"]
    # lco = lco[lco["holeType"] != "GFA"]
    apo = df[df["wokType"]=="APO"]
    # apo = apo[apo["holeType"] != "GFA"]

    # print(apo[["ix", "iy", "iz"]])

    plt.figure(figsize=(8,8))
    sns.scatterplot(x="x", y="y", hue="z", style="holeType", data=lco)
    ax = plt.gca()
    ax.axis("equal")
    plt.title("LCO")

    plt.figure(figsize=(8,8))
    sns.scatterplot(x="x", y="y", hue="z", style="holeType", data=apo)
    ax = plt.gca()
    ax.axis("equal")
    plt.title("APO")


    plt.figure(figsize=(8,8))
    x = apo["x"].to_numpy()
    y = apo["y"].to_numpy()
    uv = apo[["ix", "iy"]].to_numpy()
    uv[:,0] = uv[:,0]*10000
    nn = numpy.linalg.norm(uv, axis=1)
    # v = apo["iy"].to_numpy()
    # plt.quiver(x, y, uv[:,0]/nn, uv[:,1]/nn, angles="xy")
    plt.quiver(x, y, apo["ix"], apo["iy"], angles="xy")
    ax = plt.gca()
    ax.axis("equal")
    plt.xlabel("x")
    plt.ylabel('y')
    plt.title("local direction of $\hat{x}$")

    plt.figure(figsize=(9,9))
    for index, row in apo.iterrows():
        x = row["x"]
        y = row["y"]
        plt.plot(x,y,alpha=0)
        plt.text(x, y,
            str(row["holeID"]),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=6,
            color="black",
        )
    plt.xlabel("x")
    plt.ylabel("y")


    plt.show()

if __name__ == "__main__":
    # plotIGES()
    compileDataIGES()
    plotCompiledData()
    # plotHoleTypes()
    # plotIGES()
    # plotDirectionsEdraw()


    # compileDataIGES()

    # df = compileDataEdraw()
    # dfAPO = df[df["wokType"]=="APO"]
    # dfAPO = dfAPO[dfAPO["rWok"] > 310]
    # dfAPO = dfAPO[dfAPO["holeType"] == "Fiducial"]
    # print("nfiducials", len(dfAPO))
    # print(dfAPO[["wokType", "holeType", "rWok", "thetaWok"]])



