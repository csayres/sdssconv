from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from win32com.client import gencache
import os, time, sys
import numpy
numpy.random.seed(0)
import itertools

# Notes
#
# The python project and script was tested with the following tools:
#       Python 3.4.3 for Windows (32-bit) (https://www.python.org/downloads/) - Python interpreter
#       Python for Windows Extensions (32-bit, Python 3.4) (http://sourceforge.net/projects/pywin32/) - for COM support
#       Microsoft Visual Studio Express 2013 for Windows Desktop (https://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx) - easy-to-use IDE
#       Python Tools for Visual Studio (https://pytools.codeplex.com/) - integration into Visual Studio
#
# Note that Visual Studio and Python Tools make development easier, however this python script should should run without either installed.

class PythonStandaloneApplication(object):
    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass

    def __init__(self):
        # make sure the Python wrappers are available for the COM client and
        # interfaces
        gencache.EnsureModule('{EA433010-2BAC-43C4-857C-7AEAC4A8CCE0}', 0, 1, 0)
        gencache.EnsureModule('{F66684D7-AAFE-4A62-9156-FF7A7853F764}', 0, 1, 0)
        # Note - the above can also be accomplished using 'makepy.py' in the
        # following directory:
        #      {PythonEnv}\Lib\site-packages\wind32com\client\
        # Also note that the generate wrappers do not get refreshed when the
        # COM library changes.
        # To refresh the wrappers, you can manually delete everything in the
        # cache directory:
        #	   {PythonEnv}\Lib\site-packages\win32com\gen_py\*.*

        self.TheConnection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")
        if self.TheConnection is None:
            raise PythonStandaloneApplication.ConnectionException("Unable to intialize COM connection to ZOSAPI")

        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException("Unable to acquire ZOSAPI application")

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise PythonStandaloneApplication.LicenseException("License is not valid for ZOSAPI use")

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None

        self.TheConnection = None

    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.LoadFile(filepath, saveIfNeeded)

    def CloseFile(self, save):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException("Unable to acquire ZOSAPI application")

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if self.TheApplication.LicenseStatus is constants.LicenseStatusType_PremiumEdition:
            return "Premium"
        elif self.TheApplication.LicenseStatus is constants.LicenseStatusType_ProfessionalEdition:
            return "Professional"
        elif self.TheApplication.LicenseStatus is constants.LicenseStatusType_StandardEdition:
            return "Standard"
        else:
            return "Invalid"


if __name__ == '__main__':
    zosapi = PythonStandaloneApplication()
    value = zosapi.ExampleConstants()

    # written with ZOSAPI 16.5, 20161019, MRH

    # Set up primary optical system
    TheSystem = zosapi.TheSystem
    TheApplication = zosapi.TheApplication
    sampleDir = TheApplication.SamplesDir

    # creates a new API directory

    #! [e03s01_py]
    # Open file
    testFile = os.getcwd() + '\\duPont_SDSSV-1o1a_UW01.zmx'
    if not os.path.exists(testFile):
        # closes connection to ZOS; normally done at end of scriApt
        del zosapi
        zosapi = None
        sys.exit('Could Not Find Model')
    TheSystem.LoadFile(testFile, False)
    testFile2 = os.getcwd() + '\\junk.zmx'
    TheSystem.SaveAs(testFile2)

    TheSystemData = TheSystem.SystemData
    outfilename = "denseLCO.txt"
    if os.path.exists(outfilename):
        with open(outfilename, "r") as f:
            lastLine = f.readlines()[-1]
        startInd = int(lastLine.split(",")[0]) + 1
    else:
        with open(outfilename, "w") as f:
            f.write("# index (int), wavelength (um), xField (degrees), yField (degrees), z offset (mm, increases towards wok), yCentroid (mm), xCentroid (mm), GENC (um), SSRMS (mm)\n")
        startInd = 0


    # uniform sampling on a disk
    maxFieldAngle = 1.1
    ys = numpy.linspace(0,maxFieldAngle,maxFieldAngle*3600*2)
    xs = numpy.zeros(len(ys))
    coords = numpy.array([xs,ys]).T
    #wls = [.6231, .5400, 1.6600]
    wls = [.5400, 1.6600]
    combos = list(itertools.product(wls, coords))
    totalIters = len(combos)

    for ii, (wl, (xField,yField)) in enumerate(combos):
        if ii < startInd:
            # skip
            continue

        print("on iter %i of %i"%(ii, totalIters))
        TheSystemData.Wavelengths.GetWavelength(1).Wavelength = wl
        field1 = TheSystemData.Fields.GetField(1)
        field1.X = xField
        field1.Y = yField



        TheMFE = TheSystem.MFE
        NumOperands = TheMFE.NumberOfOperands
        TheMFE.RemoveOperandsAt(1,NumOperands)
        OptWizard = TheMFE.SEQOptimizationWizard
        OptWizard.Data = 1
        baseTool = CastTo(OptWizard, 'IWizard')
        baseTool.Apply()

        # remove the radial symmetry checkbox on wizard
        # this was the way we discovered how to do it.
        TheMFE.RemoveOperandAt(3)

        # blank operand was necessary too...
        BlankOperand = TheMFE.AddOperand()

        LocalOpt = TheSystem.Tools.OpenLocalOptimization()
        LocalOptCast = CastTo(LocalOpt,'ISystemTool')
        LocalOptCast.RunAndWaitForCompletion()
        LocalOptCast.Close()

        NormFieldOperand = TheMFE.GetOperandAt(10)
        Hx = NormFieldOperand.GetOperandCell(4)
        Hy = NormFieldOperand.GetOperandCell(5)
        OperandY = TheMFE.AddOperand()
        OperandY.ChangeType(constants.MeritOperandType_CEHY)
        OperandY.GetOperandCell(7).IntegerValue = 3
        OperandX = TheMFE.AddOperand()
        OperandX.ChangeType(constants.MeritOperandType_CEHX)
        OperandX.GetOperandCell(7).IntegerValue = 3
        OperandFWHM = TheMFE.AddOperand()
        OperandFWHM.ChangeType(constants.MeritOperandType_GENC)
        OperandFWHM.GetOperandCell(7).IntegerValue = 1
        OperandFWHM.GetOperandCell(5).DoubleValue = 0.802
        OperandFWHM.GetOperandCell(2).IntegerValue = 3
        OperandRMSSpot = TheMFE.AddOperand()
        OperandRMSSpot.ChangeType(constants.MeritOperandType_RSCE)
        OperandRMSSpot.GetOperandCell(4).DoubleValue = Hx
        OperandRMSSpot.GetOperandCell(5).DoubleValue = Hy
        OperandRMSSpot.GetOperandCell(2).IntegerValue = 3

        LocalOpt = TheSystem.Tools.OpenLocalOptimization()
        LocalOptCast = CastTo(LocalOpt,'ISystemTool')
        LocalOptCast.RunAndWaitForCompletion()
        LocalOptCast.Close()


        TheLDE = TheSystem.LDE
        Surface14 = TheLDE.GetSurfaceAt(14)

        yCentroidValue = OperandY.GetOperandCell(12).DoubleValue
        xCentroidValue = OperandX.GetOperandCell(12).DoubleValue
        fwhmValue = OperandFWHM.GetOperandCell(12).DoubleValue
        # yCentroidValue = float(yCentroid.__str__())
        zOffset = Surface14.Thickness
        RMSSpot = OperandRMSSpot.GetOperandCell(12).DoubleValue
        with open(outfilename, "a") as f:
            f.write("%i,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n"%(ii, float(wl), float(xField), float(yField), float(zOffset), float(yCentroidValue), float(xCentroidValue), float(fwhmValue), float(RMSSpot)))

    #Save and close
    TheSystem.Save()
    print("success")

    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # this until you need to.
    del zosapi
    zosapi = None



