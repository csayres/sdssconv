import numpy
import pandas as pd
from coordio import zernike
from skimage.transform import SimilarityTransform
import matplotlib.pyplot as plt

apoDF = pd.read_csv("allapo.csv", index_col=0)

# fiducials used for training
trainSet = apoDF[apoDF.fiducial == True]
trainMeas = trainSet[["fvcX", "fvcY"]].to_numpy()
trainExpect = trainSet[["fpX", "fpY"]].to_numpy()

# other points used for testing
# these are slightly within the max radius of the fiducials
testSet = apoDF[apoDF.fiducial == False]
testMeas = testSet[["fvcX", "fvcY"]].to_numpy()
testExpect = testSet[["fpX", "fpY"]].to_numpy()


# first fits trans/rot/scale transform to roughly put
# CCD xy (pixels) into xy focal plane (mm)
transRotScaleModel = SimilarityTransform()
transRotScaleModel.estimate(trainMeas, trainExpect)

# apply trans/rot/scale fit
trainSimFit = transRotScaleModel(trainMeas)
testSimFit = transRotScaleModel(testMeas)

# zernike radial order selected from cross validation from previous analysis
zernRadOrder = 8

# zernikes are defined only on unit disk
# determine max radius of inputs to scale by
max1 = numpy.max(numpy.linalg.norm(trainSimFit, axis=1))
max2 = numpy.max(numpy.linalg.norm(testSimFit, axis=1))
maxRadFP = numpy.max([max1, max2])

zf = zernike.ZernFit(
    trainSimFit[:, 0], trainSimFit[:, 1], trainExpect[:, 0], trainExpect[:, 1],
    orders=zernRadOrder, method="grad", scaleR=maxRadFP
)

# fitting is done, now apply trans/rot/scale fit to test data


# determine zernike "residual" adjustment to add to the testSimFit
xFit, yFit = zf.apply(testSimFit[:,0], testSimFit[:,1])

xResid = testExpect[:,0] - xFit
yResid = testExpect[:,1] - yFit
totalResid = numpy.sqrt(xResid**2 + yResid**2)
print("mean total resid", numpy.mean(totalResid))

fig, axs = plt.subplots(1,3, squeeze=True, figsize=(15,4))

# labels = ["xResid (true-pred)", "yResid (true-pred)", "xyResid sqrt(xResid^2+yResid^2)"]
labels = ["colored by x residual (um) ", "colored by y residual (um) ", "color by residual magnitude (um)"]
i = 0
vmin = -10
vmax = 10
cmap = "RdBu"
for label, resid in zip(labels, [xResid,yResid,totalResid]):
    # plt.figure()
    ax = axs[i]
    ax.set_title(label)
    im = ax.scatter(testExpect[:, 0], testExpect[:, 1], s=1, c=resid*1000, vmin=vmin, vmax=vmax, cmap=cmap)
    # ax.colorbar(label="micron")
    ax.axis("equal")
    ax.set_xlabel("x wok (mm)")
    if i == 0:
        ax.set_ylabel("y wok (mm)")
    plt.colorbar(im, ax=ax)
    if i == 1:
        vmin = 0
        cmap = "RdPu"
    i += 1

plt.savefig("zernResids.png", dpi=350)
plt.show()























