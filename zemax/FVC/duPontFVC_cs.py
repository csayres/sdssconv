from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from win32com.client import gencache
import os
import csv
import numpy as np
import pandas as pd
import time
import dataGen
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
        #      {PythonEnv}\Lib\site-packages\win32com\client\
        # Also note that the generate wrappers do not get refreshed when the
        # COM library changes.
        # To refresh the wrappers, you can manually delete everything in the
        # cache directory:
        #      {PythonEnv}\Lib\site-packages\win32com\gen_py\*.*

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

    # Insert Code Here
    # uncomment if first time running
    # if not os.path.exists(zosapi.TheApplication.SamplesDir + "\\API\\Python"):
    # os.makedirs(zosapi.TheApplication.SamplesDir + "\\API\\Python")


    # Reads in .csv file with x/y coordinates of field points.
    TheSystem = zosapi.TheSystem
    TheSystem.LoadFile(os.getcwd() + "\\duPont FVC-1o12a-Final_cs.zmx", False)
    TheSystemData = TheSystem.SystemData
    TheMFE = TheSystem.MFE
    TheLDE = TheSystem.LDE
    TheMCE = TheSystem.MCE

    def runOne(Xdata, Ydata, outputDataFile):

        # Setup MFE = Merit Function Editor / LDE = lense Data Editor / MCE = Multi-Configuration Editor
        totalPoints = 0
        for x in Xdata:
            totalPoints += len(x)

        print("totalPoints",totalPoints)
        print("chunks",len(Xdata))

        TheSystemData.Fields.DeleteAllFields()

        # clears any residual data in Merit Function Editor (MFE)
        while TheMFE.NumberOfOperands > 1:
            TheMFE.RemoveOperandAt(1)
            if TheMFE.NumberOfOperands==2:
                TheMFE.AddOperand()
                TheMFE.RemoveOperandAt(1)

        # clears Multi-Configuration Editor (MCE) and fills with n configurations
        TheMCE.DeleteAllConfigurations()
        for i in range(len(Xdata)-1):
            TheMCE.AddConfiguration(False)

        for j in range(49):
            TheSystemData.Fields.AddField(0,0,1)

        # fills MCE with points, must have after fields are filled
        for j in range(len(Xdata[0])):
            TheMCE.AddOperand()
            TheMCE.GetOperandAt(3*j+2).ChangeType(constants.MultiConfigOperandType_XFIE)
            TheMCE.GetOperandAt(3*j+2).Param1 = j
            for k in range(len(Xdata)):
                if j>len(Xdata[k])-1:
                    break
                TheMCE.GetOperandAt(3*j+2).GetOperandCell(1+k).DoubleValue = Xdata[k][j]

            TheMCE.AddOperand()
            TheMCE.GetOperandAt(3*j+3).ChangeType(constants.MultiConfigOperandType_YFIE)
            TheMCE.GetOperandAt(3*j+3).Param1 = j
            for k in range(len(Xdata)):
                if j>len(Ydata[k])-1:
                    break
                TheMCE.GetOperandAt(3*j+3).GetOperandCell(1+k).DoubleValue = Ydata[k][j]

            TheMCE.AddOperand()

        # puts in VCX,VCY,VDX,VDY operands into MCE
        for n in range(len(Xdata[0])):
                TheMCE.InsertNewOperandAt(7*n+4)
                TheMCE.GetOperandAt(7*n+4).ChangeType(constants.MultiConfigOperandType_FVCX)
                TheMCE.GetOperandAt(7*n+4).Param1 = n
                TheMCE.InsertNewOperandAt(7*n+5)
                TheMCE.GetOperandAt(7*n+5).ChangeType(constants.MultiConfigOperandType_FVCY)
                TheMCE.GetOperandAt(7*n+5).Param1 = n
                TheMCE.InsertNewOperandAt(7*n+6)
                TheMCE.GetOperandAt(7*n+6).ChangeType(constants.MultiConfigOperandType_FVDX)
                TheMCE.GetOperandAt(7*n+6).Param1 = n
                TheMCE.InsertNewOperandAt(7*n+7)
                TheMCE.GetOperandAt(7*n+7).ChangeType(constants.MultiConfigOperandType_FVDY)
                TheMCE.GetOperandAt(7*n+7).Param1 = n

        # for each config, this sets all vignetting factors=0.9 then compputes it 4 times
        # (for accuracy ajustment), then fills the MCE operands with those values
        for k in range(len(Xdata)):
            TheMCE.SetCurrentConfiguration(k+1)
            for m in range(len(Xdata[k])):
                TheSystemData.Fields.GetField(m+1).VCX = 0.9
                TheSystemData.Fields.GetField(m+1).VCY = 0.9
                TheSystemData.Fields.GetField(m+1).VDX = 0.9
                TheSystemData.Fields.GetField(m+1).VDY = 0.9

            TheSystemData.Fields.SetVignetting()
            TheSystemData.Fields.SetVignetting()
            TheSystemData.Fields.SetVignetting()
            TheSystemData.Fields.SetVignetting()

            VCX = []
            VCY = []
            VDX = []
            VDY = []

            for q in range(len(Xdata[k])):
                VCX.append(TheSystemData.Fields.GetField(q+1).VCX)
                VCY.append(TheSystemData.Fields.GetField(q+1).VCY)
                VDX.append(TheSystemData.Fields.GetField(q+1).VDX)
                VDY.append(TheSystemData.Fields.GetField(q+1).VDY)

            for p in range(len(Xdata[k])):
                # colby says to hand modify these
                # TheMCE.GetOperandAt(7*p+4).GetOperandCell(k+1).DoubleValue = VCX[p]
                # TheMCE.GetOperandAt(7*p+5).GetOperandCell(k+1).DoubleValue = VCY[p]
                TheMCE.GetOperandAt(7*p+4).GetOperandCell(k+1).DoubleValue = 0.98
                TheMCE.GetOperandAt(7*p+5).GetOperandCell(k+1).DoubleValue = 0.98
                TheMCE.GetOperandAt(7*p+6).GetOperandCell(k+1).DoubleValue = VDX[p]
                TheMCE.GetOperandAt(7*p+7).GetOperandCell(k+1).DoubleValue = VDY[p]


        #### Inputs Here #####
        configs = TheMCE.NumberOfConfigurations
        #points per configuration
        points = TheSystemData.Fields.NumberOfFields
        SampleSize = 80
        DENC_Sample_Size = 3
        DENC_I_Sample_Size = 3
        ImageSurface = TheLDE.NumberOfSurfaces - 1

        print("configs",configs)

        # adds a cenx and ceny operand for each point
        # surface = 24, wave = 1, pol = 0, weight = 1
        # other are what are specified in Inputs


        for j in range(len(Xdata)):
            TheMCE.SetCurrentConfiguration(j+1)
            for i in range(points):

                x = 50*2*j+2*i+3
                y = 50*2*j+2*i+4

                TheMFE.AddOperand()
                TheMFE.GetOperandAt((x)).ChangeType(constants.MeritOperandType_CENX)
                CastTo(TheMFE.GetOperandAt(x), 'IEditorRow').GetCellAt(2).IntegerValue = ImageSurface
                CastTo(TheMFE.GetOperandAt(x), 'IEditorRow').GetCellAt(3).IntegerValue = 1
                CastTo(TheMFE.GetOperandAt(x), 'IEditorRow').GetCellAt(4).IntegerValue = i+1
                CastTo(TheMFE.GetOperandAt(x), 'IEditorRow').GetCellAt(6).IntegerValue = SampleSize
                CastTo(TheMFE.GetOperandAt(x), 'IEditorRow').GetCellAt(11).DoubleValue = 0

                TheMFE.AddOperand()
                TheMFE.GetOperandAt(y).ChangeType(constants.MeritOperandType_CENY)
                CastTo(TheMFE.GetOperandAt(y), 'IEditorRow').GetCellAt(2).IntegerValue = ImageSurface
                CastTo(TheMFE.GetOperandAt(y), 'IEditorRow').GetCellAt(3).IntegerValue = 1
                CastTo(TheMFE.GetOperandAt(y), 'IEditorRow').GetCellAt(4).IntegerValue = i+1
                CastTo(TheMFE.GetOperandAt(y), 'IEditorRow').GetCellAt(6).IntegerValue = SampleSize
                CastTo(TheMFE.GetOperandAt(y), 'IEditorRow').GetCellAt(11).DoubleValue = 0

        TheMFE.RemoveOperandAt(2)
        for i in range(1,len(Xdata)):
            TheMFE.InsertNewOperandAt(len(Xdata[1])*2*i+1+i)
            TheMFE.GetOperandAt((len(Xdata[1])*2*i+1+i)).ChangeType(constants.MeritOperandType_CONF)
            CastTo(TheMFE.GetOperandAt(len(Xdata[1])*2*i+1+i), 'IEditorRow').GetCellAt(2).IntegerValue = i+1

        # calculates merit funtion for further use
        TheMFE.CalculateMeritFunction()

        # output values for centroids
        # first converts to strings then floats because of how Zemax stores data
        xVals = []
        yVals = []


        print("points",points)

        for j in range(len(Xdata)):
            #print("j",j)
            for i in range(points):
                #print("i",i)
                _xVals = (str(CastTo(TheMFE.GetOperandAt(50*2*j+2*i+2+j), 'IEditorRow').GetCellAt(12)))
                _yVals = (str(CastTo(TheMFE.GetOperandAt(50*2*j+2*i+3+j), 'IEditorRow').GetCellAt(12)))
                xVals.append((str(CastTo(TheMFE.GetOperandAt(50*2*j+2*i+2+j), 'IEditorRow').GetCellAt(12))))
                yVals.append((str(CastTo(TheMFE.GetOperandAt(50*2*j+2*i+3+j), 'IEditorRow').GetCellAt(12))))
                #print("_xVals",_xVals)
                #print("_yVals",_yVals)
                #print("")

        num_to_cut = 50*len(Xdata) - (50-len(Xdata[len(Xdata)-1]))
        xVals = xVals[0:totalPoints]
        yVals = yVals[0:totalPoints]
        for i in range(len(xVals)):
            #print("i",i,xVals[i],yVals[i],type(xVals[i]),type(yVals[i]),)
            try:
                _xVal=float(xVals[i])
                _yVal=float(yVals[i])
            except:
                _xVal=np.nan 
                _yVal=np.nan
            xVals[i]=_xVal
            yVals[i]=_yVal

        # sets up and writes original telescope focal plane corrdinates and computed centroid values to csv file
        # fpX/Y = focal plane X/Y coordinates in mm
        # fvcX/Y = fiber view camera X/Y coordinates in pixels (divided by 0.006 mm below)
        fpX = []
        fpY = []

        for i in range(len(Xdata)):
            fpX.extend(Xdata[i])
            fpY.extend(Ydata[i])

        fpX = np.array(fpX)
        fpY = np.array(fpY)

        fvcX = [xVals[i]/0.006 for i in range(len(xVals))]
        fvcY = [yVals[i]/0.006 for i in range(len(yVals))]
        fvcX = np.array(fvcX)
        fvcY = np.array(fvcY)

        print("shapes", fpX.shape, fpY.shape, fvcX.shape, fvcY.shape)

        with open(outputDataFile, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            names = np.array(['fpX','fpY','fvcX','fvcY'#,'FWHM'
                             ])
            numbers = np.array([fpX,fpY,fvcX,fvcY#,FWHM
                               ]).T
            data = np.append([names],numbers,axis=0)
            writer.writerows(data)

    tStart = time.time()
    for seed in range(50):
    #for seed in [3,4]:
        print("on %i of 50"%(seed+1))
        if seed == 0:
            # do colby's arrangement
            fileName = "lcoFVC_colby.csv"
            Xdata = dataGen.Xdata_duPont
            Ydata = dataGen.Ydata_duPont
        elif seed == 1:
            # do full hex
            fileName = "lcoFVC_allHex.csv"
            Xdata = dataGen.Xdata_allHex
            Ydata = dataGen.Ydata_allHex
        #elif seed == 2:
            #fileName = "lcoFVC_fidHex.csv"
            #Xdata = dataGen.Xdata_fidHex
            #Ydata = dataGen.Ydata_fidHex
        else:
            fileName = "lcofVC_uniform_seed_%i.csv"%seed
            Xdata, Ydata = dataGen.randomSample(seed)
        if os.path.exists(fileName):
            continue # dont redo files
        runOne(Xdata, Ydata, fileName)
    print('took %.2f minutes'%((time.time()-tStart)/60))

    #Save file
    TheSystem.SaveAs(os.getcwd() + '\\duPontjunk.zmx')

    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so for maximum performance you do not do
    # this until you need to.
    del zosapi
    zosapi = None