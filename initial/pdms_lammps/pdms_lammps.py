import os
import subprocess
import Maps
import MapsChemistryDataModel
import MapsLAMMPS
from PyQt5.QtWidgets import QMessageBox, qApp

class MapsProtocol:
  def setup(self, app, plugin, result, molecule, jobName):
    writer=MapsLAMMPS.LAMMPSWriter(app, plugin)
    fileName=str(molecule.getName())
    fileName=fileName.replace(" ", "_")
    writer.setOption("C", "5.343  10.126  0.759  0.8563  0.0  ")
    writer.setOption("CoulombPrecision", "0.0001")
    writer.setOption("CoulombSummation", "particle mesh")
    writer.setOption("H", "4.5280  13.8904  0.371  1.0698  0.0  ")
    writer.setOption("O", "8.741  13.364  0.669  0.9745  0.0  ")
    writer.setOption("PressureTensor", "no")
    writer.setOption("Qeq", "true")
    writer.setOption("QeqCutoff", "10")
    writer.setOption("QeqEvery", "1")
    writer.setOption("QeqMaxiter", "100")
    writer.setOption("QeqPrecision", "0.000001")
    writer.setOption("QeqStyle", "qeq/point")
    writer.setOption("Si", "4.168  6.974  1.176  0.7737  0.0  ")
    writer.setOption("VDWDispersion", "yes")
    writer.setOption("angleEnergy", "true")
    writer.setOption("bondEnergy", "true")
    writer.setOption("cellDimensions", "xyz")
    writer.setOption("coordinateFile", fileName + ".data")
    writer.setOption("coulombEnergy", "true")
    writer.setOption("cutoff", "12.0")
    writer.setOption("dielectric", "1.0")
    writer.setOption("endPressure", "1.000000")
    writer.setOption("energyConvergence", "0.0001")
    writer.setOption("ensemble", "NVE")
    writer.setOption("forceCalculations", "5000")
    writer.setOption("forceConvergence", "0.000001")
    writer.setOption("forceFile", fileName + ".frcdump")
    writer.setOption("improperEnergy", "true")
    writer.setOption("kineticEnergy", "true")
    writer.setOption("logFrequency", "100")
    writer.setOption("longRangeEnergy", "true")
    writer.setOption("method", "steepestDescent")
    writer.setOption("potentialEnergy", "true")
    writer.setOption("pressure", "true")
    writer.setOption("pressureDamping", "350.0")
    writer.setOption("restartAtEnd", str(molecule.getName()))
    writer.setOption("restartFile", str(molecule.getName()))
    writer.setOption("restartFrequency", "100")
    writer.setOption("scale12", "0.000000")
    writer.setOption("scale12Coulomb", "0.000000")
    writer.setOption("scale13", "0.000000")
    writer.setOption("scale13Coulomb", "0.000000")
    writer.setOption("scale14", "1.000000")
    writer.setOption("scale14Coulomb", "1.000000")
    writer.setOption("seed", "1739793307")
    writer.setOption("startPressure", "1.000000")
    writer.setOption("steps", "500")
    writer.setOption("switching", "off")
    writer.setOption("temperature", "true")
    writer.setOption("timestep", "1")
    writer.setOption("title", str(molecule.getName()))
    writer.setOption("torsionEnergy", "true")
    writer.setOption("totalEnergy", "true")
    writer.setOption("trajectoryFile", fileName + ".dump")
    writer.setOption("trajectoryFrequency", "100")
    writer.setOption("type", "forcefield")
    writer.setOption("typeOfCalculation", "optimization")
    writer.setOption("vanDerWaalsEnergy", "true")
    writer.setOption("velocityFile", fileName + ".veldump")
    success=writer.createInput(result, molecule, jobName)
    if not success:
      QMessageBox.critical(app, "Note",  writer.getErrorDescription())
    return (success)

app=Maps.MapsApp.getContainer()
app.setInteractive(False)
app.setFramesOptions(10, 0, False, False, False)
plugin=MapsLAMMPS.LAMMPSPlugin.get()
protocol=MapsProtocol()
dm=MapsChemistryDataModel.ChemistryDataModel.get()
project=dm.findProjectByName("Amorphous project")
if project == None:
  QMessageBox.critical(app, "Note", "Project \"Amorphous project\" not found.")
else:
  experiment=Maps.Experiment(dm, None, project, DataObject.END_TYPE)
  experiment.setName("LAMMPS experiment")
  project.addExperiment(experiment)
  result=Maps.ExperimentalResult(dm, experiment)
  result.setName("Results")
  experiment.addExperimentalResult(result)
  for molecule in project.getMolecules():
    if not os.path.exists(str(molecule.getName())):
      os.mkdir(str(molecule.getName()))
    os.chdir(str(molecule.getName()))
    if protocol.setup(app, plugin, result, molecule, molecule.getName()):
      if os.name == "posix":
        prefix="./"
        startup=None
      else:
        prefix=""
        # Prevent shell window from showing up
        startup=subprocess.STARTUPINFO()
        startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
      subprocess.call(["python", prefix + "run.py"], shell=False, startupinfo=startup)
      plugin.loadFile(result, molecule.getName() + ".log")
      qApp.processEvents()
    os.chdir("..")
app.setInteractive(True)
