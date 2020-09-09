import numpy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from fieldCoords import fieldAngle2Cart, cart2Sph

GFA_loc = 329 # mm
# diagonal
GFA_size = numpy.sqrt(2*(13.5/1000*2048)**2) # mm
GFA_max_r = GFA_loc + 0.5*GFA_size
GFA_min_r = GFA_loc - 0.5*GFA_size
SMALL_NUM = 1e-12 # epsilon for computational accuracy
MICRON_PER_MM = 1000

# offsets along the optical axis
# that reference the focus (thickness) zemax measurements
# to the primary mirror vertex (reported from Kal)
APO_Z_OFFSET = 691.4797888
LCO_Z_OFFSET = 993.03

"""Notes:

should investigate using dot products rather than trig
for direction cosines? may be more numerical stable?

Field directions at LCO with plugplates are inverted
with respect to APO, should make sure the XY directions
as I have defined them work at LCO or if there was something else besides
and accident that required that plug plate hack.

in tests, scrutenize SMALL_NUM and decide if it's good enough

https://drive.google.com/drive/folders/1mVJ_UASiKZ7pdUES2yCfJYVL9i-ZBmAC

rename centroid to focal for clarity?

LCO Distortion model email: Feb 18 2020

understanding plate coordinates email: May 3 2020

https://github.com/sdss/platedesign/blob/master/pro/plate/lco_rdistort.pro

https://github.com/sdss/platedesign/blob/master/pro/plate/lco_scales.pro
"""


def loadZemaxData(filePath):
    """Import data from text file produced from ZEMAX scripting model of focal plane

    Input parameter is a path to csv file with the following columns (first row is a header):

    1. index (int)
    2. wavelength (um): input wavelength used for zemax analysis
    3. xField (degrees): ZEMAX-style input field angle
    4. yField (degrees): ZEMAX-style input field angle
    5. zCentroid (mm, increases towards wok away from source):
        output z location of centroid on focal surface
        (z axis is parallel with optical path). IMPORTANT ZEMAX z
        direction points from sky through telescope, the opposite convention
        to the cartesian z direction for field definitions.  The z zeropoint
        is somewhat arbitrary in practice, the focal surface is shifted via telescope
        focus.
    6. yCentroid (mm): output y location of centroid on focal surface
    7. xCentroid (mm): output x location of centroid on focal surface
    8. GENC (um): Diameter of encircled energy at 80.2 percent (roughly a FWHM of a Gaussian)
    9. SSRMS (mm): Radial spot size RMS, another measure of PSF spread

    notes on input field angles from zemax manual:
    Positive field angles imply positive slope for the ray in that direction,
    and thus refer to **negative** coordinates on distant objects. ZEMAX converts
    x field angles ( αx ) and y field angles ( αy ) to ray direction cosines
    using the following formulas:

    tanαx = l/n
    tanαy = m/n
    l^2 + m^2 + n^2 = 1

    where l, m, and n are the x, y, and z direction cosines.

    Output dataframe column descriptions:
    ...

    Parameters
    ------------
    filePath: str
        Path to input file

    Returns
    ---------
    result: pandas.DataFrame
        Data frame constructed from input CSV file
    """
    # column outputs
    names = [
        "wavelength",  # micron
        "xField",  # degrees (ZEMAX-style)
        "yField",  # degrees (ZEMAX-style)
        "zCentroid",  # mm, increases toward sky, 0 at primary mirror vertex
        "yCentroid",  # mm -DEC due to 180 deg image rotation
        "xCentroid",  # mm -RA due to 180 degree image rotation
        "GENC",  # micron (fwhm proxy)
        "SSRMS"  # micron (fwhm proxy)
    ]
    df = pd.read_csv(filePath, names=names, skiprows=0, comment="#")

    # convert SSRMS to microns
    df["SSRMS"] = df["SSRMS"] * MICRON_PER_MM

    # reverse direction so that Z axis such that it points toward sky
    # zeropoint at primary mirror vertex
    # it is tied to the WOK coord sys when
    # stars at guide wavelength are focused on
    # center of chip (this defines z offset between the focal plane and wok)
    # this also serves to make xyz right handed
    if "APO" in filePath:
        zoff = APO_Z_OFFSET
        df["observatory"] = ["APO"]*len(df)
    elif "LCO" in filePath:
        zoff = LCO_Z_OFFSET
        df["observatory"] = ["LCO"]*len(df)
    else:
        print("warining, unknown telescope model")
        zoff = 0
    # reference the zCentroid to the M1 vertex,
    # reverse direction such that positive Z
    # points toward sky (opposite of zemax convention)
    df["zCentroid"] = -1 * (df["zCentroid"] + zoff)

    # add categorical for wavelength
    waveCat = []
    for wl in df["wavelength"]:
        if wl == 1.66:
            waveCat.append("Apogee")
        elif wl == 0.6231:
            waveCat.append("GFA")
        else:
            # .54
            waveCat.append("BOSS")

    df["waveCat"] = waveCat

    # add spherical coords from xyFields
    # phiField = []
    # thetaField = []
    # # _tf = []
    # for xField, yField in zip(df["xField"], df["yField"]):
    #     _x, _y, _z = fieldAngle2Cart(xField, yField)
    #     _phi, _theta = cart2Sph(_x, _y, _z)
    #     phiField.append(_phi)
    #     thetaField.append(_theta)
    #     # if xField == 0:
    #     #     _tf.append(_theta)

    _x, _y, _z = fieldAngle2Cart(df["xField"].to_numpy(), df["yField"].to_numpy())
    thetaField, phiField = cart2Sph(_x, _y, _z)

    df["phiField"] = phiField
    df["thetaField"] = thetaField

    # if len(_tf) > 0:
    #     _tf = numpy.array(_tf)
    #     m = numpy.mean(_tf)
    #     diff = m - _tf
    #     a = numpy.argwhere(numpy.abs(diff)>0.1)

    #     print("filepath", filePath)
    #     print("results", numpy.mean(_tf), _tf-numpy.mean(_tf))
    #     print("args", len(a), a)
    #     print("wrong", diff[a])
    #     print("xField", df["xField"].to_numpy()[a])
    #     print("yField", df["yField"].to_numpy()[a])
    #     print("phiField", df["phiField"].to_numpy()[a])
    # print()
    # print()


    # add cyclindrical coords to centroids
    thetaCentroid = []
    rCentroid = []
    for xCent, yCent in zip(df["xCentroid"], df["yCentroid"]):
        _r = numpy.linalg.norm([xCent, yCent])
        _theta = numpy.degrees(numpy.arctan2(yCent, xCent))
        if _theta < 0:
            _theta += 360
        thetaCentroid.append(_theta)
        rCentroid.append(_r)

    df["thetaCentroid"] = thetaCentroid
    df["rCentroid"] = rCentroid

    return df


class SphFit(object):
    def __init__(self, dataFrame, rMax=None):
        """Generate a spherical fit give radial and z (cylindrical coords)

        Center of sphere is taken to lie on the positive z axis, so the
        sphere is generated by revolving about the z axis

        Attributes
        -------------
        b_fit : float
            circle center position (along z axis)
        r_fit: float
            circle radius
        powers: list of integers
            powers used to create the polynomial distortion model
        distortCoeffs: list of floats
            coefficients corresponding to powers for forward distortion model
        revDistortCoeffs: list of floats
            coefficients corresponding to powers for reverse distortion model
        df : pd.DataFrame
            raw data used to create fits

        Paramters
        ------------
        dataFrame: pd.DataFrame
            output from loadZemaxData
        rMax: None or float
            maximum focal plane radius to consider when fitting
        """
        self.df = dataFrame.copy()
        self.b_fit = None
        self.r_fit = None
        self.powers = None
        self.distortCoeffs = None
        self.revDistortCoeffs = None

        if rMax is not None:
            self.df = self.df[self.df["rCentroid"] < rMax]

        self.df.reset_index()

    # def fit(self):
    #     self.sphFit()
    #     self.distortFit()

    def fitSphere(self):
        """Find the best-fit (least squares) solution for a spherical focal
        plane.

        Stores the circle center as the b_fit attribute
        Stores the circle radius as the r_fit attribute
        Appends the *fit* focal phi angle for all data used to self.df
        Appends the focal theta angle to self.df
        """

        # fit sphere parameters b_fit and r_fit
        r = self.df["rCentroid"].to_numpy()
        z = self.df["zCentroid"].to_numpy()

        A = numpy.ones((len(z), 2))
        A[:, 0] = z
        sol = numpy.linalg.lstsq(A, -r**2 - z**2, rcond=None)
        a, b = sol[0]
        self.b_fit = a / -2.  # circle center
        self.r_fit = numpy.sqrt(self.b_fit**2 - b)  # circle radius

    def predictZ(self, r):
        """Given a radial position on the focal plane predict the z (focus direction)

        Paramters
        -----------
        r: float or 1D array
            radial position

        Returns
        --------
        result: float or 1D array
            z position
        """
        if None in [self.b_fit, self.r_fit]:
            raise RuntimeError("Must call fitSphere prior to predictZ")
        return -1*(numpy.sqrt(self.r_fit**2 - r**2) - self.b_fit)

    def computeFocalItems(self):
        r = self.df["rCentroid"].to_numpy()
        z = self.df["zCentroid"].to_numpy()
        zResids = z - self.predictZ(r)
        # store z residuals
        self.df["zResiduals"] = zResids

        # generate focal phis (degree off boresight)
        # angle off-axis from optical axis
        v = self.df[["rCentroid", "zCentroid"]].to_numpy()
        v[:, 1] = v[:, 1] - self.b_fit

        # unit vector pointing toward object on focal plane from circle center
        # arc from vertex to off axis
        v = v / numpy.vstack([numpy.linalg.norm(v, axis=1)] * 2).T
        downwardZaxis = numpy.array([0, -1])  # FP lands at -Z so, Angle from sphere center towards ground
        # phi angle between ray and optical axis measured from sphere center
        phiFocal = numpy.degrees(numpy.arccos(v.dot(downwardZaxis)))
        self.df["phiFocal"] = phiFocal

        # determine azimuthal coordinate of focal plane
        thetaFocal = numpy.degrees(numpy.arctan2(self.df["yCentroid"], self.df["xCentroid"]))
        thetaFocal[thetaFocal < 0] += 360
        self.df["thetaFocal"] = thetaFocal

    def fitDistortion(self, powers):
        """Fit a polynomial distortion model for predicting focal plane phi's
        from field phi's


        Parameters
        -----------
        powers : list
            list of powers to fit, eg [0, 1, 3] will be a model of the form:
            phiFocal = co + c1*phiField + c2*phiField**3
            so 0 is the bias, 1 is the linear term, extra elements are powers.
        """
        self.powers = powers

        if self.b_fit is None:
            raise RuntimeError("Must call fitSphere prior to fitDistortion")

        phiField = self.df["phiField"].to_numpy()
        phiFocal = self.df["phiFocal"].to_numpy()

        # fit forward model (phiFocal from phiField)
        A = []
        for p in powers:
            A.append(phiField**p)

        A = numpy.array(A).T
        sol = numpy.linalg.lstsq(A, phiFocal, rcond=None)
        self.distortCoeffs = sol[0]

        # fit reverse model (phiField from phiFocal)
        A = []
        for p in powers:
            A.append(phiFocal**p)

        A = numpy.array(A).T
        sol = numpy.linalg.lstsq(A, phiField, rcond=None)
        self.revDistortCoeffs = sol[0]

    def computeDistortResid(self):
        phiField = self.df["phiField"].to_numpy()
        phiFocal = self.df["phiFocal"].to_numpy()
        distortionResiduals = phiFocal - self.predictPhiFocal(phiField)
        # convert distortionResiduals to microns tangential to focal surface
        distortionResiduals = 2 * numpy.sin(
            numpy.radians(distortionResiduals / 2)
        ) * self.r_fit * MICRON_PER_MM
        self.df["distortResid"] = distortionResiduals

    def predictPhiFocal(self, phiField):
        """Given an incoming source angle phiField determine focal plane angle
        phiFocal

        The routine uses the best fit sphere to the focal plane and the
        polynomial distortion solution.

        Paramters
        -----------
        phiField : float or array
            The incoming angle (deg) of the ray with respect to the optical axis

        Returns
        ---------
        result : float or array
            The outgoing angle(s) phiFocal of the ray with respect to the optical axis
        """
        if self.powers is None:
            raise RuntimeError("Must call fitDistortion prior to precictPhiFocal")

        try:
            terms = numpy.zeros(len(phiField))
        except:
            # not an array
            terms = 0
        for coeff, power in zip(self.distortCoeffs, self.powers):
            terms += coeff * phiField**power
        return terms


def plotRadialFocalSurface(figname, apModel, bossModel, site):

    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    wls = ["Boss", "Apogee"]
    for wl, model in zip(wls, [bossModel, apModel]):
        # plot zemax data
        r, z = model.df[["rCentroid", "zCentroid"]].to_numpy().T
        ax.plot(r, z, '.', alpha=0.1, label=wl)
        # plot best fit sphere
        rValues = numpy.linspace(0, GFA_max_r, 1000)
        zValues = model.predictZ(rValues)
        ax.plot(rValues, zValues, ':', color="black", zorder=10, linewidth=3, label="%s (fit radius: %i mm)"%(wl, int(model.r_fit)))

    ax.axvline(GFA_max_r, linestyle='--', color="red", label="GFA edge")

    # plot best fit sphere

    ax.set_xlabel("rCentroid (mm)")
    ax.set_ylabel("zCentroid (mm)")
    ax.set_title("%s %s radial focal surface fits"%(site, figname))
    ax.legend()
    plt.savefig("%s_%s_radFocSurf.png"%(figname, site), dpi=350)
    plt.close()


def plotXYFocalSurface(figname, apModel, bossModel, site):
    thetas = numpy.linspace(0, numpy.pi*2, 1000)
    xGFA = GFA_max_r*numpy.cos(thetas)
    yGFA = GFA_max_r*numpy.sin(thetas)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13, 6))
    fig.suptitle("%s %s focal surface"%(site, figname))
    sns.scatterplot(x="xCentroid", y="yCentroid", hue="zCentroid", data=apModel.df, linewidth=0, alpha=0.5, ax=ax1)
    ax1.set_ylabel("yCentroid (mm)")
    ax1.set_xlabel("xCentroid (mm)")
    ax1.axis("equal")
    ax1.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax1.legend()
    ax1.set_title("Apogee")

    sns.scatterplot(x="xCentroid", y="yCentroid", hue="zCentroid", data=bossModel.df, linewidth=0, alpha=0.5, ax=ax2)
    ax2.set_ylabel("yCentroid (mm)")
    ax2.set_xlabel("xCentroid (mm)")
    ax2.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax2.legend()
    ax2.axis("equal")
    ax2.set_title("BOSS")

    plt.savefig("%s_%s_xyFocSurf.png"%(figname, site), dpi=350)
    plt.close()


def plotXYFocalSurfaceResid(figname, apModel, bossModel, site):
    thetas = numpy.linspace(0, numpy.pi*2, 1000)
    xGFA = GFA_max_r*numpy.cos(thetas)
    yGFA = GFA_max_r*numpy.sin(thetas)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13, 6))
    fig.suptitle("%s %s focal surface spherical residuals"%(site, figname))
    sns.scatterplot(x="xCentroid", y="yCentroid", hue="zResiduals", data=apModel.df, linewidth=0, alpha=0.5, ax=ax1)
    ax1.set_ylabel("yCentroid (mm)")
    ax1.set_xlabel("xCentroid (mm)")
    ax1.axis("equal")
    ax1.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax1.legend()
    ax1.set_title("Apogee")

    sns.scatterplot(x="xCentroid", y="yCentroid", hue="zResiduals", data=bossModel.df, linewidth=0, alpha=0.5, ax=ax2)
    ax2.set_ylabel("yCentroid (mm)")
    ax2.set_xlabel("xCentroid (mm)")
    ax2.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax2.legend()
    ax2.axis("equal")
    ax2.set_title("BOSS")

    plt.savefig("%s_%s_xyFocSurfResid.png"%(figname, site), dpi=350)
    plt.close()


def plotRadialFocalSurfaceResid(figname, apModel, bossModel, site):

    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    wls = ["Boss", "Apogee"]
    for wl, model in zip(wls, [bossModel, apModel]):
        # plot zemax data
        r, z = model.df[["rCentroid", "zResiduals"]].to_numpy().T
        rms = numpy.sqrt(numpy.mean(z**2)) * MICRON_PER_MM
        ax.plot(r, z, '.', alpha=0.5, label="%s (RMS: %.2e micron)"%(wl, rms))

    ax.axvline(GFA_max_r, linestyle='--', color="red", label="GFA edge")

    # plot best fit sphere

    ax.set_xlabel("rCentroid (mm)")
    ax.set_ylabel("zResiduals (mm)")
    ax.set_title("%s %s radial focal surface fit residuals"%(site, figname))
    ax.legend()
    plt.savefig("%s_%s_radFocSurfResid.png"%(figname, site), dpi=350)
    plt.close()


def plotDistortionStack(figname, apModel, bossModel, site):

    distortionTerms = [
        [1],
        [1, 3],
        [1, 3, 5],
        [1, 3, 5, 7],
        [1, 3, 5, 7, 9],
        [1, 3, 5, 7, 9, 11],
    ]
    fig, axs = plt.subplots(len(distortionTerms), 2, figsize=(10, 7))
    pltInd = 0
    # bins = numpy.linspace(-1, 1, 50)
    for d in distortionTerms:
        apModel.fitDistortion(d)
        bossModel.fitDistortion(d)
        apModel.computeDistortResid()
        bossModel.computeDistortResid()
        ax = axs[pltInd]
        apR, apDR = apModel.df[["rCentroid", "distortResid"]].to_numpy().T
        bossR, bossDR = bossModel.df[["rCentroid", "distortResid"]].to_numpy().T
        apRMS = numpy.sqrt(numpy.mean(apDR**2))
        bossRMS = numpy.sqrt(numpy.mean(bossDR**2))

        ax[0].plot(apR, apDR, '.', alpha=0.05, label="RMS: %.2e um\norders: %s"%(apRMS, str(d)))
        ax[1].plot(bossR, bossDR, '.', alpha=0.05, label="RMS: %.2e um\norders: %s"%(bossRMS, str(d)))

        if pltInd == 0:
            ax[0].set_title("Apogee")
            ax[1].set_title("Boss")
        ax[0].legend()
        ax[1].legend()
        if pltInd == len(axs)-1:
            ax[0].set_xlabel("rCentroid (mm)")
            ax[1].set_xlabel("rCentroid (mm)")
        else:
            ax[0].xaxis.set_ticklabels([])
            ax[1].xaxis.set_ticklabels([])
        if pltInd == int(len(axs)/2):
            ax[0].set_ylabel("distortResid (um)")
        # ax[0].set_ylabel("rCentroid (mm)")
        pltInd += 1
    fig.suptitle("%s %s\nodd polynomial distortion fit residuals"%(site, figname))
    plt.savefig("%s_%s_polyOrders.png"%(figname, site), dpi=350)
    plt.close()


def plotXYDistortionResiduals(figname, apModel, bossModel, site):
    apR, apDR = apModel.df[["rCentroid", "distortResid"]].to_numpy().T
    bossR, bossDR = bossModel.df[["rCentroid", "distortResid"]].to_numpy().T
    apRMS = numpy.sqrt(numpy.mean(apDR**2))
    bossRMS = numpy.sqrt(numpy.mean(bossDR**2))

    thetas = numpy.linspace(0, numpy.pi*2, 1000)
    xGFA = GFA_max_r*numpy.cos(thetas)
    yGFA = GFA_max_r*numpy.sin(thetas)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13, 6))
    fig.suptitle("%s %s radial distortion residuals (um)\nApogee RMS: %.2e -- Boss RMS: %.2e\norders: %s"%(site, figname, apRMS, bossRMS, apModel.powers))
    sns.scatterplot(x="xCentroid", y="yCentroid", hue="distortResid", data=apModel.df, linewidth=0, alpha=0.5, ax=ax1)
    ax1.set_ylabel("yCentroid (mm)")
    ax1.set_xlabel("xCentroid (mm)")
    ax1.axis("equal")
    ax1.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax1.set_title("Apogee")

    sns.scatterplot(x="xCentroid", y="yCentroid", hue="distortResid", data=bossModel.df, linewidth=0, alpha=0.5, ax=ax2)
    ax2.set_ylabel("yCentroid (mm)")
    ax2.set_xlabel("xCentroid (mm)")
    ax2.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax2.axis("equal")
    ax2.set_title("BOSS")

    plt.savefig("%s_%s_xyDistortResid.png"%(figname, site), dpi=350)
    plt.close()

def plotRadFWHM(figname, df, site):

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9, 9))
    fig.suptitle("%s %s"%(figname, site))
    sns.scatterplot(x="rCentroid", y="SSRMS", hue="waveCat", data=df, ax=ax1, alpha=0.1)
    sns.scatterplot(x="rCentroid", y="GENC", hue="waveCat", data=df, ax=ax2, alpha=0.1)
    plt.savefig("%s_%s_fwhm.png"%(figname, site), dpi=350)
    plt.close()


def plotXYFWHM(figname, df, site):

    thetas = numpy.linspace(0, numpy.pi*2, 1000)
    xGFA = GFA_max_r*numpy.cos(thetas)
    yGFA = GFA_max_r*numpy.sin(thetas)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(9,9))

    fig.suptitle("%s %s"%(figname, site))
    sns.scatterplot(x="xCentroid", y="yCentroid", hue="SSRMS", data=df[df["waveCat"] == "Apogee"], linewidth=0, alpha=0.5, ax=ax1)
    ax1.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    sns.scatterplot(x="xCentroid", y="yCentroid", hue="SSRMS", data=df[df["waveCat"] == "BOSS"], linewidth=0, alpha=0.5, ax=ax2)
    ax2.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    sns.scatterplot(x="xCentroid", y="yCentroid", hue="GENC", data=df[df["waveCat"] == "Apogee"], linewidth=0, alpha=0.5, ax=ax3)
    ax3.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    sns.scatterplot(x="xCentroid", y="yCentroid", hue="GENC", data=df[df["waveCat"] == "BOSS"], linewidth=0, alpha=0.5, ax=ax4)
    ax4.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax1.set_title("Apogee")
    ax2.set_title("Boss")
    plt.savefig("%s_%s_XYfwhm.png"%(figname, site), dpi=350)
    plt.close()


def joseRegions():
    gfaLCOradius = 333.42
    gfaAPOradius = 333.09
    gfaTheta = [30, 90, 150, 210, 270, 330]
    _GFA_size = 27.6

    fileBase = os.path.join(os.getenv("SDSSCONV_DIR"), "data")

    for obs, gfar in zip(["LCO", "APO"], [gfaLCOradius, gfaAPOradius]):
        for sampling in ["uniformGFA"]:
            filename = fileBase + "/" + sampling + obs + ".txt"
            fullThetas = []
            minRs = []
            maxRs = []
            for t in gfaTheta:
                df = loadZemaxData(filename)
                df = df[df["rCentroid"] < gfar+0.5*_GFA_size]
                df = df[df["rCentroid"] > gfar-0.5*_GFA_size]
                dtheta = numpy.degrees(numpy.arctan2(0.5*_GFA_size, gfar))
                df = df[df["thetaCentroid"] < t+dtheta]
                df = df[df["thetaCentroid"] > t-dtheta]
                fullThetas.append(numpy.max(df["thetaField"])-numpy.min(df["thetaField"]))
                minRs.append(numpy.min(df["phiField"]))
                maxRs.append(numpy.max(df["phiField"]))

                #print(obs, t, numpy.min(df["phiField"]), numpy.max(df["phiField"]), numpy.min(df["thetaField"]), numpy.max(df["thetaField"]))
            print(obs, numpy.mean(minRs), numpy.mean(maxRs), numpy.mean(fullThetas))


def compileZemaxData():
    """Take the individual uniform simulation runs from zemax, compile them into a single
    pandas style csv
    """
    fileBase = os.path.join(os.getenv("SDSSCONV_DIR"), "data")
    dfs = []
    for obs in ["LCO", "APO"]:
        for sampling in ["uniform", "uniformGFA"]:
            filename = fileBase + "/" + sampling + obs + ".txt"
            df = loadZemaxData(filename)
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(fileBase + "/zemaxData.csv", index=False)


def generateFits():
    """ Generate spherical and distortion fits from the uniform data
    for each wavelengths at each site

    For now all wavelenghts are fit over the whole focal plane to just
    beyond the GFA outer corner.  In the future we should probably
    model the GFA wavelength only around the GFA region to minimize
    systematic offsets from reality...

    Probably want to continue using full field GFA fits for the FSC

    may want to consider lowering the order of the fits
    """

    fileBase = os.path.join(os.getenv("SDSSCONV_DIR"), "data")

    df = pd.read_csv(fileBase + "/zemaxData.csv")
    # truncate the data 2mm beyond the outer edge of the GFA
    df = df[df["rCentroid"] < GFA_max_r + 2]
    for obs in ["LCO", "APO"]:
        _df = df[df["observatory"] == obs]
        if obs == "LCO":
            # fit LCO with a 7th order
            powers = [1, 3, 5, 7]
        else:
            # fit apo with a 9th order
            powers = [1, 3, 5, 7, 9]
        for waveCat in ["Apogee", "BOSS", "GFA"]:
            sph = SphFit(_df[_df["waveCat"] == waveCat])
            sph.fitSphere()
            sph.computeFocalItems()
            sph.fitDistortion(powers)
            fwdModel = ["%.5e(phiField)^%i"%(c,p) for c,p in zip(sph.distortCoeffs, sph.powers)]
            fwdModel = " + ".join(fwdModel)

            revModel = ["%.5e(phiFocal)^%i"%(c,p) for c,p in zip(sph.revDistortCoeffs, sph.powers)]
            revModel = " + ".join(revModel)
            print("%s %s r=%i b=%i"%(obs, waveCat, sph.r_fit, sph.b_fit))
            print("phiFocal = %s"%(fwdModel))
            print("phiField = %s"%(revModel))
            print("")
            print("")



    # import pdb; pdb.set_trace()

    # for obs in ["LCO", "APO"]:
    #     for sampling in ["uniform"]: #["uniform", "reticle", "dense"]:
    #         # use dense sampling to fit sphere and distortion
    #         filename = fileBase + "/" + sampling + obs + ".txt"
    #         df = loadZemaxData(filename)
    #         # print(obs, sampling, len(df))
    #         # remove the GFA wavelength, clutters visualization and doesn't add much
    #         df = df[df["waveCat"] != "GFA"]

    #         # df = df[df["rCentroid"] < 300]

    #         # print(obs, sampling)
    #         # for index, row in list(df.iterrows())[:100]:
    #         #     xf, yf, rCen = row[["xField", "yField", "rCentroid"]]
    #         #     print(rCen, xf, yf, index)

    #         # fit spheres
    #         sphAp = SphFit(df[df["waveCat"] == "Apogee"])
    #         sphAp.fitSphere()
    #         print("ap points", len(sphAp.df))
    #         sphAp.computeFocalItems()
    #         sphBoss = SphFit(df[df["waveCat"] == "BOSS"])
    #         print("boss points", len(sphBoss.df))
    #         sphBoss.fitSphere()
    #         sphBoss.computeFocalItems()

    #         plotRadialFocalSurface(sampling, sphAp, sphBoss, obs)
    #         plotRadialFocalSurfaceResid(sampling, sphAp, sphBoss, obs)
    #         plotDistortionStack(sampling, sphAp, sphBoss, obs)
    #         plotRadFWHM(sampling, df, obs)

    #         if sampling == "uniform":
    #             if obs == "LCO":
    #                 powers = [1,3,5,7]
    #             else:
    #                 powers = [1,3,5,7,9]
    #             sphAp.fitDistortion(powers)
    #             print("%s apogee sph r=%i b=%i"%(obs, int(sphAp.r_fit), int(sphAp.b_fit)))
    #             print("%s apogee coeffs %s"%(obs, ",".join(["%.4e"%x for x in sphAp.distortCoeffs])))
    #             sphAp.computeDistortResid()
    #             sphBoss.fitDistortion(powers)
    #             print("%s boss sph r=%i b=%i"%(obs, int(sphBoss.r_fit), int(sphBoss.b_fit)))
    #             print("%s boss coeffs %s"%(obs, ",".join(["%.4e"%x for x in sphBoss.distortCoeffs])))
    #             sphBoss.computeDistortResid()
    #             plotXYFocalSurface(sampling, sphAp, sphBoss, obs)
    #             plotXYFocalSurfaceResid(sampling, sphAp, sphBoss, obs)
    #             plotXYDistortionResiduals(sampling, sphAp, sphBoss, obs)
    #             plotRadFWHM(sampling, df, obs)
    #             plotXYFWHM(sampling, df, obs)

    # plt.show()

if __name__ == "__main__":
    compileZemaxData()
    generateFits()

