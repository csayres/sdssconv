import numpy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from fieldCoords import fieldAngle2Cart, cart2Sph

GFA_loc = 329 # mm
GFA_size = 13.5/1000*2048 # mm
GFA_max_r = GFA_loc + 0.5*GFA_size
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
    8. DENC (um): Diameter of encircled energy at 80.2 percent (roughly a FWHM of a Gaussian)
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
        "DENC",  # micron (fwhm proxy)
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
    elif "LCO" in filePath:
        zoff = LCO_Z_OFFSET
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
    phiField = []
    thetaField = []
    # _tf = []
    for xField, yField in zip(df["xField"], df["yField"]):
        _cartXYZ = fieldAngle2Cart([xField, yField])
        _phi, _theta = cart2Sph(_cartXYZ)
        phiField.append(_phi)
        thetaField.append(_theta)
        # if xField == 0:
        #     _tf.append(_theta)
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

        if self.b_fit is None:
            raise RuntimeError("Must call fitSphere prior to fitDistortion")

        phiField = self.df["phiField"].to_numpy()
        phiFocal = self.df["phiFocal"].to_numpy()
        A = []
        for p in powers:
            A.append(phiField**p)

        A = numpy.array(A).T
        sol = numpy.linalg.lstsq(A, phiFocal, rcond=None)
        self.distortCoeffs = sol[0]
        self.powers = powers

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

    ax.set_xlabel("centroid r (mm)")
    ax.set_ylabel("centroid z (mm)")
    ax.set_title("%s %s radial focal surface fits"%(site, figname))
    ax.legend()
    plt.savefig("%s_%s_radFocSurf.png"%(figname, site), dpi=350)
    plt.close()

def plotXYFocalSurface(figname, apModel, bossModel, site):
    thetas = numpy.linspace(0, numpy.pi*2, 1000)
    xGFA = GFA_max_r*numpy.cos(thetas)
    yGFA = GFA_max_r*numpy.sin(thetas)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13, 6))
    fig.suptitle(site + " focal surface")
    sns.scatterplot(x="xCentroid", y="yCentroid", hue="zCentroid", data=apModel.df, linewidth=0, alpha=0.5, ax=ax1)
    ax1.set_ylabel("y centroid (mm)")
    ax1.set_xlabel("x centroid (mm)")
    ax1.axis("equal")
    ax1.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax1.set_title("Apogee")

    sns.scatterplot(x="xCentroid", y="yCentroid", hue="zCentroid", data=apModel.df, linewidth=0, alpha=0.5, ax=ax2)
    ax2.set_ylabel("y centroid (mm)")
    ax2.set_xlabel("x centroid (mm)")
    ax2.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax2.axis("equal")
    ax2.set_title("BOSS")

    plt.savefig("%s_%s_xyFocSurf.png"%(figname, site), dpi=350)
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

    ax.set_xlabel("centroid r (mm)")
    ax.set_ylabel("z residual (mm)")
    ax.set_title("%s %s radial focal surface fit residuals"%(site, figname))
    ax.legend()
    plt.savefig("%s_%s_radFocSurfResid.png"%(figname, site), dpi=350)
    plt.close()

def plotXYFocalSurface(figname, apModel, bossModel, site):
    thetas = numpy.linspace(0, numpy.pi*2, 1000)
    xGFA = GFA_max_r*numpy.cos(thetas)
    yGFA = GFA_max_r*numpy.sin(thetas)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13, 6))
    fig.suptitle(site + " focal surface spherical residuals")
    sns.scatterplot(x="xCentroid", y="yCentroid", hue="zResiduals", data=apModel.df, linewidth=0, alpha=0.5, ax=ax1)
    ax1.set_ylabel("y centroid (mm)")
    ax1.set_xlabel("x centroid (mm)")
    ax1.axis("equal")
    ax1.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax1.set_title("Apogee")

    sns.scatterplot(x="xCentroid", y="yCentroid", hue="zResiduals", data=apModel.df, linewidth=0, alpha=0.5, ax=ax2)
    ax2.set_ylabel("y centroid (mm)")
    ax2.set_xlabel("x centroid (mm)")
    ax2.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax2.axis("equal")
    ax2.set_title("BOSS")

    plt.savefig("%s_%s_xyFocSurfResid.png"%(figname, site), dpi=350)
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
            ax[0].set_xlabel("centroid r (mm)")
            ax[1].set_xlabel("centroid r (mm)")
        else:
            ax[0].xaxis.set_ticklabels([])
            ax[1].xaxis.set_ticklabels([])
        if pltInd == int(len(axs)/2):
            ax[0].set_ylabel("distort residual (um)")
        # ax[0].set_ylabel("centroid r (mm)")
        pltInd += 1
    fig.suptitle(obs + "\nodd polynomial distortion fit residuals")
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
    fig.suptitle(site + " radial distortion residuals (um)\nApogee RMS: %.2e -- Boss RMS: %.2e\norders: %s"%(apRMS, bossRMS, apModel.powers))
    sns.scatterplot(x="xCentroid", y="yCentroid", hue="distortResid", data=apModel.df, linewidth=0, alpha=0.5, ax=ax1)
    ax1.set_ylabel("y centroid (mm)")
    ax1.set_xlabel("x centroid (mm)")
    ax1.axis("equal")
    ax1.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax1.set_title("Apogee")

    sns.scatterplot(x="xCentroid", y="yCentroid", hue="distortResid", data=apModel.df, linewidth=0, alpha=0.5, ax=ax2)
    ax2.set_ylabel("y centroid (mm)")
    ax2.set_xlabel("x centroid (mm)")
    ax2.plot(xGFA, yGFA, '--', color="red", label="GFA edge")
    ax2.axis("equal")
    ax2.set_title("BOSS")

    plt.savefig("%s_%s_xyDistortResid.png"%(figname, site), dpi=350)
    plt.close()


if __name__ == "__main__":
    fileBase = os.path.join(os.getenv("SDSSCONV_DIR"), "zemax/Python-Zemax-Files")

    for obs in ["APO", "LCO"]:
        for sampling in ["dense", "uniform"]:
            # use dense sampling to fit sphere and distortion
            filename = fileBase + "/" + sampling + obs + ".txt"
            df = loadZemaxData(filename)
            # remove the GFA wavelength, clutters visualization and doesn't add much
            df = df[df["waveCat"] != "GFA"]
            # truncate the data 2mm beyond the outer edge of the GFA
            # df = df[df["rCentroid"] < GFA_max_r + 2]
            df = df[df["rCentroid"] < 300]

            # fit spheres
            sphAp = SphFit(df[df["waveCat"] == "Apogee"])
            sphAp.fitSphere()
            sphAp.computeFocalItems()
            sphBoss = SphFit(df[df["waveCat"] == "BOSS"])
            sphBoss.fitSphere()
            sphBoss.computeFocalItems()

            plotRadialFocalSurface(sampling, sphAp, sphBoss, obs)
            plotRadialFocalSurfaceResid(sampling, sphAp, sphBoss, obs)
            plotDistortionStack(sampling, sphAp, sphBoss, obs)

            if sampling == "uniform":
                if obs == "LCO":
                    powers = [1,3,5]
                else:
                    powers = [1,3,5,9]
                sphAp.fitDistortion(powers)
                sphAp.computeDistortResid()
                sphBoss.fitDistortion(powers)
                sphBoss.computeDistortResid()
                plotXYFocalSurface(sampling, sphAp, sphBoss, obs)
                plotXYDistortionResiduals(sampling, sphAp, sphBoss, obs)



            continue
        # fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,9))
        # apRMS = numpy.sqrt(numpy.sum(sphAp.df["zResiduals"]**2)/len(sphAp.df["zResiduals"]))*MICRON_PER_MM
        # bossRMS = numpy.sqrt(numpy.sum(sphBoss.df["zResiduals"]**2)/len(sphBoss.df["zResiduals"]))*MICRON_PER_MM
        # sns.scatterplot(x="rCentroid", y="zResiduals", data=sphAp.df, linewidth=0, alpha=0.5, ax=ax1)
        # yPos = 0.5*numpy.max(sphAp.df["zResiduals"])
        # ax1.text(10, yPos, "dZ RMS: %.2e um"%apRMS)
        # ax1.set_ylabel("Apogee z residuals (mm)")
        # ax1.set_title(obs + " z residuals")
        # sns.scatterplot(x="rCentroid", y="zResiduals", data=sphBoss.df, linewidth=0, alpha=0.5, ax=ax2)
        # ax2.set_ylabel("Boss z residuals (mm)")
        # yPos = 0.5*numpy.max(sphBoss.df["zResiduals"])
        # ax2.text(10, 0, "dZ RMS: %.2e um"%bossRMS)
        # plt.savefig(obs+"_dense_sphResid.png", dpi=350)
        # plt.close()

        # fit distortions
        # distortionTerms = [
        #     [1, 3],
        #     [1, 3, 5],
        #     [1, 3, 5, 7],
        #     [1, 3, 5, 7, 9],
        #     [1, 3, 5, 7, 9, 11]
        # ]
        # fig, axs = plt.subplots(len(distortionTerms), 2, figsize=(10, 7))
        # pltInd = 0
        # bins = numpy.linspace(-1, 1, 50)
        # for d in distortionTerms:
        #     sphAp.fitDistortion(d)
        #     sphBoss.fitDistortion(d)
        #     sphAp.computeDistortResid()
        #     sphBoss.computeDistortResid()
        #     ax = axs[pltInd]
        #     apHist = numpy.copy(sphAp.df["distortResid"].to_numpy())
        #     bossHist = numpy.copy(sphBoss.df["distortResid"].to_numpy())
        #     apRMS = numpy.sqrt(numpy.sum(apHist**2)/len(apHist))
        #     bossRMS = numpy.sqrt(numpy.sum(bossHist**2)/len(bossHist))

        #     ax[0].hist(apHist, bins, label="RMS: %.2e um"%(apRMS))
        #     ax[1].hist(bossHist, bins, label="RMS: %.2e um"%(bossRMS))

        #     if pltInd == 0:
        #         ax[0].set_title("Apogee")
        #         ax[1].set_title("Boss")
        #     ax[0].legend()
        #     ax[1].legend()
        #     if pltInd != len(axs)-1:
        #         ax[0].xaxis.set_ticklabels([])
        #         ax[1].xaxis.set_ticklabels([])
        #     else:
        #         ax[0].set_xlabel("distortion residual (um)")
        #         ax[1].set_xlabel("distortion residual (um)")
        #     ax[0].set_ylabel("order %i"%d[-1])
        #     ax[1].yaxis.set_ticklabels([])
        #     ax[0].yaxis.set_ticklabels([])
        #     pltInd += 1
        # fig.suptitle(obs + "\nodd polynomial distortion fit")
        # plt.savefig(obs+"_polyOrders.png", dpi=350)
        # plt.close()

        # refit to favorite orders
        # if obs == "APO":
        #     apPowers = [1,3,5,7]
        #     bossPowers = apPowers
        #     sphAp.fitDistortion(apPowers)
        #     sphBoss.fitDistortion(bossPowers)
        # else:
        #     apPowers = [1,3,5,7,9]
        #     bossPowers = [1,3,5,7,9]
        #     sphAp.fitDistortion(apPowers)
        #     sphBoss.fitDistortion(bossPowers)

        # sphAp.computeDistortResid()
        # sphBoss.computeDistortResid()

        # apRMS = numpy.sqrt(numpy.sum(sphAp.df["distortResid"]**2)/len(sphAp.df["distortResid"]))
        # bossRMS = numpy.sqrt(numpy.sum(sphBoss.df["distortResid"]**2)/len(sphBoss.df["distortResid"]))

        # g = sns.jointplot(x="rCentroid", y="distortResid", data=sphAp.df, kind="kde")
        # g.fig.suptitle(obs + " Apogee order %i\nRMS: %.2e um"%(apPowers[-1], apRMS))
        # g.set_axis_labels("rCentroid (mm)", "distortion residual (um)")
        # plt.savefig(obs+"_apogee_selectedPoly_kde.png", dpi=350)
        # plt.close()
        # # g.fig.set_size_inches(9,9)

        # # fig, ax = plt.subplots(1,1)
        # g = sns.jointplot(x="rCentroid", y="distortResid", data=sphBoss.df, kind="kde")
        # g.fig.suptitle(obs + " BOSS order %i\nRMS: %.2e um"%(bossPowers[-1], bossRMS))
        # g.set_axis_labels("rCentroid (mm)", "distortion residual (um)")
        # plt.savefig(obs+"_boss_selectedPoly_kde.png", dpi=350)
        # plt.close()

        # g = sns.jointplot(x="rCentroid", y="distortResid", data=sphAp.df, alpha=0.5)
        # g.fig.suptitle(obs + " Apogee order %i\nRMS: %.2e um"%(apPowers[-1], apRMS))
        # g.set_axis_labels("rCentroid (mm)", "distortion residual (um)")
        # plt.savefig(obs+"_apogee_selectedPoly_hist.png", dpi=350)
        # plt.close()
        # # g.fig.set_size_inches(9,9)

        # # fig, ax = plt.subplots(1,1)
        # g = sns.jointplot(x="rCentroid", y="distortResid", data=sphBoss.df, alpha=0.5)
        # g.fig.suptitle(obs + " BOSS order %i\nRMS: %.2e um"%(bossPowers[-1], bossRMS))
        # g.set_axis_labels("rCentroid (mm)", "distortion residual (um)")
        # plt.savefig(obs+"_boss_selectedPoly_hist.png", dpi=350)
        # plt.close()


        # ################### check against uniform distributions
        # #######################################################
        # sampling = "uniform"
        # filename = fileBase + "/" + sampling + obs + ".txt"
        # df = loadZemaxData(filename)
        # # remove the GFA wavelength, clutters visualization and doesn't add much
        # df = df[df["waveCat"] != "GFA"]
        # # truncate the data 2mm beyond the outer edge of the GFA
        # df = df[df["rCentroid"] < GFA_max_r + 2]

        # fig, ax = plt.subplots(1,1)
        # sns.lineplot(x="phiField", y="SSRMS", hue="waveCat", data=df, ax=ax)
        # fig.suptitle(obs)
        # fig, ax = plt.subplots(1,1)
        # sns.lineplot(x="phiField", y="DENC", hue="waveCat", data=df, ax=ax)
        # fig.suptitle(obs)

        # if obs == "APO":
        #     fig, ax = plt.subplots(1,1)
        #     sns.scatterplot(x="xField", y="yField", hue="SSRMS", data=df[df["waveCat"]=="BOSS"], ax=ax)
        #     fig, ax = plt.subplots(1,1)
        #     sns.scatterplot(x="xField", y="yField", hue="DENC", data=df[df["waveCat"]=="BOSS"], ax=ax)

        # # copy best fit parameters from dense into uniform
        # sphApUnif = SphFit(df[df["waveCat"] == "Apogee"])
        # sphApUnif.r_fit = sphAp.r_fit
        # sphApUnif.b_fit = sphAp.b_fit
        # sphApUnif.powers = sphAp.powers
        # sphApUnif.distortCoeffs = sphAp.distortCoeffs
        # sphApUnif.computeFocalItems()
        # sphApUnif.computeDistortResid()


        # sphBossUnif = SphFit(df[df["waveCat"] == "BOSS"])
        # sphBossUnif.r_fit = sphBoss.r_fit
        # sphBossUnif.b_fit = sphBoss.b_fit
        # sphBossUnif.powers = sphBoss.powers
        # sphBossUnif.distortCoeffs = sphBoss.distortCoeffs
        # sphBossUnif.computeFocalItems()
        # sphBossUnif.computeDistortResid()

        # # visualize fits
        # fig, ax = plt.subplots(1,1, figsize=(13,7))
        # # sns.scatterplot(x="rCentroid", y="zCentroid", hue="waveCat", data=df, ax=ax, linewidth=0, alpha=0.5)
        # sns.lineplot(x="rCentroid", y="zCentroid", hue="waveCat", data=df, ax=ax, alpha=0.5)
        # ax.axvline(GFA_max_r, linestyle='--', color="red", label="GFA")

        # # plot best fit sphere
        # rValues = numpy.linspace(0, GFA_max_r, 1000)
        # zValues = sphAp.predictZ(rValues)
        # ax.plot(rValues, zValues, ':', color="black", label="Ap fit radius: %i mm"%(int(sphAp.r_fit)))

        # zValues = sphBoss.predictZ(rValues)
        # ax.plot(rValues, zValues, ':', color="black", label="Boss fit radius: %i mm"%(int(sphBoss.r_fit)))

        # ax.set_xlabel("centroid r (mm)")
        # ax.set_ylabel("centroid z (mm)")
        # ax.set_title(obs + " uniform focal plane curvature")
        # ax.legend()
        # plt.savefig(obs+"_test_sphFit.png", dpi=350)
        # plt.close()

        # # plot spherical fit residuals
        # fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,9))
        # apRMS = numpy.sqrt(numpy.sum(sphApUnif.df["zResiduals"]**2)/len(sphApUnif.df["zResiduals"]))*MICRON_PER_MM
        # bossRMS = numpy.sqrt(numpy.sum(sphBossUnif.df["zResiduals"]**2)/len(sphBossUnif.df["zResiduals"]))*MICRON_PER_MM
        # sns.scatterplot(x="rCentroid", y="zResiduals", data=sphApUnif.df, linewidth=0, alpha=0.5, ax=ax1)
        # ax1.set_ylabel("Apogee z residuals (mm)")
        # ax1.set_title(obs + " uniform z residuals")
        # sns.scatterplot(x="rCentroid", y="zResiduals", data=sphBossUnif.df, linewidth=0, alpha=0.5, ax=ax2)
        # ax2.set_ylabel("Boss z residuals (mm)")
        # yPos = 0.5*numpy.max(sphApUnif.df["zResiduals"])
        # ax1.text(10, yPos, "dZ RMS: %.2e um"%apRMS)
        # yPos = 0.5*numpy.max(sphBossUnif.df["zResiduals"])
        # ax2.text(10, yPos, "dZ RMS: %.2e um"%bossRMS)
        # plt.savefig(obs+"_test_sphResid.png", dpi=350)
        # plt.close()

        # # 3D plots
        # thetas = numpy.linspace(0, numpy.pi*2, 1000)
        # xGFA = GFA_max_r*numpy.cos(thetas)
        # yGFA = GFA_max_r*numpy.sin(thetas)
        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10, 10))
        # fig.suptitle(obs + " uniform")
        # sns.scatterplot(x="xCentroid", y="yCentroid", hue="zCentroid", data=sphApUnif.df, linewidth=0, alpha=0.5, ax=ax1)
        # ax1.set_ylabel("y centroid (mm)")
        # ax1.set_xlabel("x centroid (mm)")
        # ax1.plot(xGFA, yGFA, '--', color="red", label="GFA")
        # ax1.set_title("Apogee")

        # sns.scatterplot(x="xCentroid", y="yCentroid", hue="zCentroid", data=sphBossUnif.df, linewidth=0, alpha=0.5, ax=ax2)
        # ax2.set_ylabel("y centroid (mm)")
        # ax2.set_xlabel("x centroid (mm)")
        # ax2.plot(xGFA, yGFA, '--', color="red", label="GFA")
        # ax2.set_title("BOSS")

        # # fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,7))
        # sns.scatterplot(x="xCentroid", y="yCentroid", hue="zResiduals", data=sphApUnif.df, linewidth=0, alpha=0.5, ax=ax3)
        # ax3.set_ylabel("y centroid (mm)")
        # ax3.set_xlabel("x centroid (mm)")
        # ax3.plot(xGFA, yGFA, '--', color="red", label="GFA")
        # ax3.set_title("Apogee - sph fit")

        # sns.scatterplot(x="xCentroid", y="yCentroid", hue="zResiduals", data=sphBossUnif.df, linewidth=0, alpha=0.5, ax=ax4)
        # ax4.set_ylabel("y centroid (mm)")
        # ax4.set_xlabel("x centroid (mm)")
        # ax4.plot(xGFA, yGFA, '--', color="red", label="GFA")
        # ax4.set_title("BOSS - sph fit")
        # plt.savefig(obs+"_test_sphResid_3D.png", dpi=350)
        # plt.close()

        # apRMS = numpy.sqrt(numpy.sum(sphApUnif.df["distortResid"]**2)/len(sphApUnif.df["distortResid"]))
        # bossRMS = numpy.sqrt(numpy.sum(sphBossUnif.df["distortResid"]**2)/len(sphBossUnif.df["distortResid"]))

        # g = sns.jointplot(x="rCentroid", y="distortResid", data=sphApUnif.df, kind="kde")
        # g.fig.suptitle(obs + " uniform Apogee order %i\nRMS: %.2e um"%(apPowers[-1], apRMS))
        # g.set_axis_labels("rCentroid (mm)", "distortion residual (um)")
        # plt.savefig(obs+"_apogee_test_distortResid_kde.png", dpi=350)
        # plt.close()
        # # g.fig.set_size_inches(9,9)

        # # fig, ax = plt.subplots(1,1)
        # g = sns.jointplot(x="rCentroid", y="distortResid", data=sphBossUnif.df, kind="kde")
        # g.fig.suptitle(obs + " uniform BOSS order %i\nRMS: %.2e um"%(bossPowers[-1], bossRMS))
        # g.set_axis_labels("rCentroid (mm)", "distortion residual (um)")
        # plt.savefig(obs+"_boss_test_distortResid_kde.png", dpi=350)
        # plt.close()

        # g = sns.jointplot(x="rCentroid", y="distortResid", data=sphApUnif.df, alpha=0.5)
        # g.fig.suptitle(obs + " uniform Apogee order %i\nRMS: %.2e um"%(apPowers[-1], apRMS))
        # g.set_axis_labels("rCentroid (mm)", "distortion residual (um)")
        # plt.savefig(obs+"_apogee_test_distortResid_hist.png", dpi=350)
        # plt.close()
        # # g.fig.set_size_inches(9,9)

        # # fig, ax = plt.subplots(1,1)
        # g = sns.jointplot(x="rCentroid", y="distortResid", data=sphBossUnif.df, alpha=0.5)
        # g.fig.suptitle(obs + " uniform BOSS order %i\nRMS: %.2e um"%(bossPowers[-1], bossRMS))
        # g.set_axis_labels("rCentroid (mm)", "distortion residual (um)")
        # plt.savefig(obs+"_boss_test_distortResid_hist.png", dpi=350)
        # plt.close()

        # import pdb; pdb.set_trace()


    plt.show()


    def tonsoplots():
        ###### ANALYSIS #####
        for obs in ["APO", "LCO"]:
            for sampling in ["dense", "uniform"]:
                filename = fileBase + "/" + sampling + obs + ".txt"
                df = loadZemaxData(filename)
                # remove the GFA wavelength, clutters visualization and doesn't add much
                df = df[df["waveCat"] != "GFA"]
                # truncate the data 2mm beyond the outer edge of the GFA
                df = df[df["rCentroid"] < GFA_max_r + 2]
                title = obs + " " + sampling
                print(title + " len pts", len(df["waveCat"]=="Apogee"))

                thetas = numpy.linspace(0, numpy.pi*2, 1000)
                xGFA = GFA_max_r*numpy.cos(thetas)
                yGFA = GFA_max_r*numpy.sin(thetas)
                # field inputs
                # fig, ax = plt.subplots(1,1, figsize=(9,9))
                # sns.scatterplot(x="xField", y="yField", data=df, ax=ax, linewidth=0, alpha=0.5)
                # ax.set_title(title + " field inputs")
                # ax.legend()
                # ax.set_xlabel("xField (deg)")
                # ax.set_ylabel("yField (deg)")

                # focal plane outputs
                # fig, ax = plt.subplots(1,1, figsize=(9,9))
                # sns.scatterplot(x="xCentroid", y="yCentroid", hue="waveCat", data=df, ax=ax, linewidth=0, alpha=0.5)
                # ax.plot(xGFA, yGFA, '--', color="red", label="GFA")
                # ax.set_xlabel("centroid x (mm)")
                # ax.set_ylabel("centroid y (mm)")
                # ax.set_title(title + " focal plane outputs")
                # ax.legend()

                # image quality plots
                # fig, ax = plt.subplots(1,1, figsize=(13,7))
                # # sns.scatterplot(x="rCentroid", y="zCentroid", hue="waveCat", data=df, ax=ax, linewidth=0, alpha=0.5)
                # sns.lineplot(x="rCentroid", y="SSRMS", hue="waveCat", data=df, ax=ax, alpha=0.5)
                # ax.axvline(GFA_max_r, linestyle='--', color="red", label="GFA")
                # ax.set_xlabel("centroid r (mm)")
                # ax.set_ylabel("spot size rms (um)")
                # ax.set_title(title + " image quality")
                # ax.legend()

                # fig, ax = plt.subplots(1,1, figsize=(13,7))
                # # sns.scatterplot(x="rCentroid", y="zCentroid", hue="waveCat", data=df, ax=ax, linewidth=0, alpha=0.5)
                # sns.lineplot(x="rCentroid", y="DENC", hue="waveCat", data=df, ax=ax, alpha=0.5)
                # ax.axvline(GFA_max_r, linestyle='--', color="red", label="GFA")
                # ax.set_xlabel("centroid r (mm)")
                # ax.set_ylabel("diameter encircled energy (um)")
                # ax.set_title(title + " image quality")
                # ax.legend()

                # radial plots, focal surface stuff
                sphAp = SphFit(df[df["waveCat"]=="Apogee"])
                sphAp.fitSphere()
                # print(title + " Apogee radius: %.2f"%sphAp.r_fit)
                sphBoss = SphFit(df[df["waveCat"]=="BOSS"])
                sphBoss.fitSphere()
                # print(title + " Boss radius: %.2f"%sphBoss.r_fit)
                # calculate raduis of focal plane curvature
                fig, ax = plt.subplots(1,1, figsize=(13,7))
                # sns.scatterplot(x="rCentroid", y="zCentroid", hue="waveCat", data=df, ax=ax, linewidth=0, alpha=0.5)
                sns.lineplot(x="rCentroid", y="zCentroid", hue="waveCat", data=df, ax=ax, alpha=0.5)
                ax.axvline(GFA_max_r, linestyle='--', color="red", label="GFA")

                # plot best fit sphere
                rValues = numpy.linspace(0, GFA_max_r, 1000)
                zValues = sphAp.predictZ(rValues)
                ax.plot(rValues, zValues, ':', color="black", label="Ap fit: %i mm"%(int(sphAp.r_fit)))

                zValues = sphBoss.predictZ(rValues)
                ax.plot(rValues, zValues, ':', color="black", label="Boss fit: %i mm"%(int(sphBoss.r_fit)))

                ax.set_xlabel("centroid r (mm)")
                ax.set_ylabel("centroid z (mm)")
                ax.set_title(title + " focal plane curvature")
                ax.legend()

                # residual focal surface plots
                fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,9))
                sns.scatterplot(x="rCentroid", y="zResiduals", data=sphAp.df, linewidth=0, alpha=0.5, ax=ax1)
                ax1.set_ylabel("Apogee z residuals (mm)")
                ax1.set_title(title + " z residuals")
                sns.scatterplot(x="rCentroid", y="zResiduals", data=sphBoss.df, linewidth=0, alpha=0.5, ax=ax2)
                ax2.set_ylabel("Boss z residuals (mm)")

                # 3D plots
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10, 10))
                fig.suptitle(title)
                sns.scatterplot(x="xCentroid", y="yCentroid", hue="zCentroid", data=sphAp.df, linewidth=0, alpha=0.5, ax=ax1)
                ax1.set_ylabel("y centroid (mm)")
                ax1.set_xlabel("x centroid (mm)")
                ax1.plot(xGFA, yGFA, '--', color="red", label="GFA")
                ax1.set_title("Apogee")

                sns.scatterplot(x="xCentroid", y="yCentroid", hue="zCentroid", data=sphBoss.df, linewidth=0, alpha=0.5, ax=ax2)
                ax2.set_ylabel("y centroid (mm)")
                ax2.set_xlabel("x centroid (mm)")
                ax2.plot(xGFA, yGFA, '--', color="red", label="GFA")
                ax2.set_title("BOSS")

                # fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,7))
                sns.scatterplot(x="xCentroid", y="yCentroid", hue="zResiduals", data=sphAp.df, linewidth=0, alpha=0.5, ax=ax3)
                ax3.set_ylabel("y centroid (mm)")
                ax3.set_xlabel("x centroid (mm)")
                ax3.plot(xGFA, yGFA, '--', color="red", label="GFA")
                ax3.set_title("Apogee - sph fit")

                sns.scatterplot(x="xCentroid", y="yCentroid", hue="zResiduals", data=sphBoss.df, linewidth=0, alpha=0.5, ax=ax4)
                ax4.set_ylabel("y centroid (mm)")
                ax4.set_xlabel("x centroid (mm)")
                ax4.plot(xGFA, yGFA, '--', color="red", label="GFA")
                ax4.set_title("BOSS - sph fit")

                # look at radial distortions
                distortionTerms = [
                    [1, 3],
                    [1, 3, 5],
                    [1, 3, 5, 7],
                    [1, 3, 5, 7, 9]
                ]
                apogeeHists = []
                bossHists = []
                apogeeCoeffs = []
                bossCoeffs = []
                for d in distortionTerms:
                    sphAp.fitDistortion(d)
                    sphBoss.fitDistortion(d)
                    # fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,9))
                    # fig.suptitle(title + "\ndistortion terms %s"%str(d))
                    # sns.scatterplot(x="rCentroid", y="distortResid", data=sphAp.df, linewidth=0, alpha=0.5, ax=ax1)
                    # ax1.set_title("Apogee")
                    # ax1.set_xlabel("")
                    # ax1.set_ylabel("distortion residuals (micron)")
                    # sns.scatterplot(x="rCentroid", y="distortResid", data=sphBoss.df, linewidth=0, alpha=0.5, ax=ax2)
                    # ax2.set_ylabel("distortion residuals (micron)")
                    # ax2.set_title("BOSS")
                    apogeeHists.append(numpy.copy(sphAp.df["distortResid"].to_numpy()))
                    bossHists.append(numpy.copy(sphBoss.df["distortResid"].to_numpy()))
                    apogeeCoeffs.append(numpy.copy(sphAp.distortCoeffs))
                    bossCoeffs.append(numpy.copy(sphBoss.distortCoeffs))


                    # fig, ax = plt.subplots(1,1, figsize=(9,9))
                    g = sns.jointplot(x="rCentroid", y="distortResid", data=sphAp.df, kind="kde")
                    g.fig.suptitle(title + "\nApogee -- %s"%str(d))
                    # g.fig.set_size_inches(9,9)

                    # fig, ax = plt.subplots(1,1)
                    g = sns.jointplot(x="rCentroid", y="distortResid", data=sphBoss.df, kind="kde")
                    g.fig.suptitle(title + "\nBOSS -- %s"%str(d))
                    # g.fig.set_size_inches(9,9)

                    g = sns.jointplot(x="rCentroid", y="distortResid", data=sphAp.df)
                    g.fig.suptitle(title + "\nApogee -- %s"%str(d))
                    # g.fig.set_size_inches(9,9)

                    # fig, ax = plt.subplots(1,1)
                    g = sns.jointplot(x="rCentroid", y="distortResid", data=sphBoss.df)
                    g.fig.suptitle(title + "\nBOSS -- %s"%str(d))
                    # g.fig.set_size_inches(9,9)

                    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14, 6.5))
                    fig.suptitle(title + "\ndistortion terms %s"%str(d))
                    sns.scatterplot(x="xCentroid", y="yCentroid", hue="distortResid", data=sphAp.df, linewidth=0, alpha=0.5, ax=ax1)
                    ax1.set_ylabel("y centroid (mm)")
                    ax1.set_xlabel("x centroid (mm)")
                    ax1.plot(xGFA, yGFA, '--', color="red", label="GFA")
                    ax1.set_title("Apogee")

                    sns.scatterplot(x="xCentroid", y="yCentroid", hue="distortResid", data=sphBoss.df, linewidth=0, alpha=0.5, ax=ax2)
                    ax2.set_ylabel("y centroid (mm)")
                    ax2.set_xlabel("x centroid (mm)")
                    ax2.plot(xGFA, yGFA, '--', color="red", label="GFA")
                    ax2.set_title("BOSS")

                fig, axs = plt.subplots(len(distortionTerms), 2, figsize=(13, 7))
                pltInd = 0
                bins = numpy.linspace(-2, 2, 50)
                for ax, apHist, bossHist, apCoeff, bossCoeff in zip(axs, apogeeHists, bossHists, apogeeCoeffs, bossCoeffs):
                    apRMS = numpy.sqrt(numpy.sum(apHist**2)/len(apHist))
                    bossRMS = numpy.sqrt(numpy.sum(bossHist**2)/len(bossHist))
                    ax[0].hist(apHist, bins, label="RMS: %.4e um"%apRMS + "\ncoeffs: "+", ".join(["%.4e"%x for x in apCoeff]))
                    ax[1].hist(bossHist, bins, label="RMS: %.4e um"%bossRMS + "\ncoeffs: "+", ".join(["%.4e"%x for x in bossCoeff]))

                    if pltInd == 0:
                        ax[0].set_title("Apogee")
                        ax[1].set_title("Boss")
                    ax[0].legend()
                    ax[1].legend()
                    if pltInd != len(axs)-1:
                        ax[0].xaxis.set_ticklabels([])
                        ax[1].xaxis.set_ticklabels([])
                    else:
                        ax[0].set_xlabel("distortResid (um)")
                        ax[1].set_xlabel("distortResid (um)")

                    pltInd += 1
                fig.suptitle(title)

        plt.show()
