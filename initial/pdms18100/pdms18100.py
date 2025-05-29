import os
import subprocess
import Maps
import MapsChemistryDataModel
import MapsGROMACS
from PyQt5.QtWidgets import QMessageBox, qApp

class MapsProtocol:
  def setup(self, app, plugin, result, molecule, jobName):
    writer=MapsGROMACS.GROMACSWriter(app, plugin)
    fileName=str(molecule.getName())
    fileName=fileName.replace(" ", "_")
    writer.setOption("DispCorr", "no")
    writer.setOption("DispCorrComment", "   ; apply long range dispersion corrections")
    writer.setOption("FourierSpaceOption", "true")
    writer.setOption("PMEOrderOption", "true")
    writer.setOption("Verlet", "true")
    writer.setOption("VerletBufferChecked", "true")
    writer.setOption("binTab", "0.002")
    writer.setOption("combRule", "2")
    writer.setOption("compressed-x-precision", "1000")
    writer.setOption("compressed-x-precisionComment", "   ; precision for compressed trajectory")
    writer.setOption("coordinateFile", fileName + ".data")
    writer.setOption("coulomb-modifier", "None")
    writer.setOption("coulombtype", "PME")
    writer.setOption("coulombtypeComment", "   ; method for electrostatics")
    writer.setOption("cutoff-scheme", "Verlet")
    writer.setOption("cutoff-schemeComment", "   ; particle based cut-offs")
    writer.setOption("emstep", "0.01")
    writer.setOption("emstepComment", "   ; initial step-size")
    writer.setOption("emtol", "10")
    writer.setOption("emtolComment", "   ; converge when maximum force is smaller than this value")
    writer.setOption("ensemble", "NVT")
    writer.setOption("epsilon-r", "1")
    writer.setOption("epsilon-rComment", "   ; Relative dielectric constant for the medium")
    writer.setOption("epsilon-surface", "0")
    writer.setOption("epsilon-surfaceComment", "   ; relative permittivity of surface")
    writer.setOption("ewald-geometry", "3d")
    writer.setOption("fourierspacing", "0.25")
    writer.setOption("fourierspacingComment", "   ; grid spacing for FFT")
    writer.setOption("fudgeLJ", "1.000000")
    writer.setOption("fudgeQQ", "1.000000")
    writer.setOption("gromacsVersion", "2021")
    writer.setOption("integrator", "steep")
    writer.setOption("integratorComment", "   ; steepest descent")
    writer.setOption("nrexcl", "3")
    writer.setOption("nstcalcenergy", "0")
    writer.setOption("nstcalcenergyComment", "   ; calculate energies every this many steps")
    writer.setOption("nstenergy", "0")
    writer.setOption("nstenergyComment", "   ; write energies to energy file")
    writer.setOption("nsteps", "1000")
    writer.setOption("nstepsComment", "   ; number of steps")
    writer.setOption("nstfout", "0")
    writer.setOption("nstfoutComment", "   ; write forces to output trajectory")
    writer.setOption("nstlist", "10")
    writer.setOption("nstlistComment", "   ; frequency to update neighbour list")
    writer.setOption("nstlog", "1000")
    writer.setOption("nstlogComment", "   ; write energies to the log file")
    writer.setOption("nstvout", "0")
    writer.setOption("nstvoutComment", "   ; write velocities to output trajectory")
    writer.setOption("nstxout", "1000")
    writer.setOption("nstxout-compressed", "0")
    writer.setOption("nstxout-compressedComment", "   ; write coordinates to compressed trajectory (.xtc file)")
    writer.setOption("nstxoutComment", "   ; write coordinates to output trajectory")
    writer.setOption("pbc", "xyz")
    writer.setOption("pbcComment", "   ; periodic boundary conditions")
    writer.setOption("periodic-molecules", "no")
    writer.setOption("pme-order", "4")
    writer.setOption("pme-orderComment", "   ; interpolation order for PME")
    writer.setOption("rcoulomb", "1.2")
    writer.setOption("rcoulombComment", "   ; short-range electrostatic cutoff (nm)")
    writer.setOption("rvdw", "1.2")
    writer.setOption("rvdwComment", "   ; short-range van der Waals cutoff (nm)")
    writer.setOption("title", str(molecule.getName()))
    writer.setOption("typeOfCalculation", "minimization")
    writer.setOption("vdw-modifier", "None")
    writer.setOption("vdwtype", "Cut-off")
    writer.setOption("vdwtypeComment", "   ; method for van der Waals")
    writer.setOption("verlet-buffer-tolerance", "0.005")
    writer.setOption("verlet-buffer-toleranceComment", "   ; allowed energy drift due to Verlet buffer in kJ/mol/ps per atom")
    success=writer.createInput(result, molecule, jobName)
    if not success:
      QMessageBox.critical(app, "Note",  writer.getErrorDescription())
    return (success)

app=Maps.MapsApp.getContainer()
app.setInteractive(False)
app.setFramesOptions(10, 0, False, False, False)
plugin=MapsGROMACS.GROMACSPlugin.get()
protocol=MapsProtocol()
dm=MapsChemistryDataModel.ChemistryDataModel.get()
project=dm.findProjectByName("Amorphous project")
if project == None:
  QMessageBox.critical(app, "Note", "Project \"Amorphous project\" not found.")
else:
  experiment=Maps.Experiment(dm, None, project, DataObject.END_TYPE)
  experiment.setName("GROMACS experiment")
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
