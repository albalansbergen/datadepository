#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import sys

def writeCommandLog(message,mode="a"):
  file=open("runinfo", mode)
  file.write(message+"\n")
  file.close()

def writeStatus(message):
  file=open("status", "w")
  file.write(message)
  file.close()

def findInPath(file):
  if os.path.isabs(file):
    return (file)
  for dir in os.environ["PATH"].split(os.path.pathsep):
    path=dir + os.path.sep + file
    if os.path.exists(path):
      return (path)

def checkGPUCapabilities():
  gpuNameApp="gpuname"
  if os.name != "posix":
    try:
      from win32com.shell import shell, shellcon
      filename = shell.SHGetFolderPath(0, shellcon.CSIDL_PROGRAM_FILES_COMMON, 0, 0) + "\Scienomics\Utils\gpuname.exe"
      if os.path.isfile( filename): gpuNameApp = filename
    except:
      writeCommandLog("error : could not locate gpuname from core")
  try:
    proc=subprocess.Popen([gpuNameApp, "-c"], shell=False, stdout=subprocess.PIPE)
    cuda=proc.stdout.readline()
  except:
    writeCommandLog("error : gpuname execution failed ("+gpuNameApp+")")
    return (False, "","")
  omp=""
  writeCommandLog("info (cuda) : "+cuda.strip())
  if len(cuda) == 0:
    proc=subprocess.Popen([gpuNameApp, "-c", "-l"], shell=False, stdout=subprocess.PIPE)
    omp=proc.stdout.readline()
    writeCommandLog("info (omp) : "+omp.strip())
  return ( len(cuda) > 0 or len(omp) > 0, cuda, omp.strip())

def getLammpsPreferences():
  arch=""
  mode=""
  preffilename = "lammps_pref"
  if os.name == "posix":
    preffile=os.path.join(os.getenv("HOME"), ".maps", preffilename)
  else:
    preffile=os.path.join(os.environ["USERPROFILE"], ".maps", preffilename)
  if preffile and os.path.isfile(preffile) and os.access(preffile, os.R_OK):
    file=open(preffile,"r")
    for line in file.readlines():
      if line.startswith("#"):
        continue
      fields = line.strip().split()
      if len(fields) == 2:
        if fields[0].startswith("arch"):
          arch = fields[1]
        elif fields[0].startswith("mode"):
          mode = fields[1]
    file.close()
  return (arch, mode)

def getExecutableName(cuda, omp, arch):
  if len(arch) == 0:
    if len(cuda) == 0:
      if len(omp) > 0:
        arch="_opencl"
    else:
      cudaVersion=cuda.split(":")[1].strip().replace(".","")
      if int(cudaVersion) > 86:
        cudaVersion=86
      arch="_cuda"
  executable="lammps"+arch
  return ( executable)

def setGPUKeywords(inputFile):
  supportGPU = True
  writeCommandLog("info : set GPU keywords")
  inFile=open(inputFile, "r")
  outFile=open(inputFile + ".gpu", "w")
  i=0
  for line in inFile.readlines():
    i=i+1
    if i == 2:
      outFile.write("package        gpu 1\n")
      outFile.write("suffix         gpu\n")
    if line.find("package") == -1 and line.find("suffix") == -1:
      outFile.write(line)
    if line.find("pair_style") != -1 and line.find("reax") != -1:
      writeCommandLog("info : non GPU supported pair style (reax)")
      supportGPU=False
  inFile.close()
  outFile.close()
  os.remove(inputFile)
  os.rename(inputFile + ".gpu", inputFile)
  return supportGPU

def removeGPUKeywords(inputFile):
  writeCommandLog("info : remove GPU keywords")
  inFile=open(inputFile, "r")
  outFile=open(inputFile + ".nogpu", "w")
  for line in inFile.readlines():
    if line.find("package") == -1 and line.find("suffix") == -1:
      outFile.write(line)
  inFile.close()
  outFile.close()
  os.remove(inputFile)
  os.rename(inputFile + ".nogpu", inputFile)

def handleSignals(signalNumber, frame=None):
  writeStatus("Killed")
  sys.exit(1)

def checkEnvironment(command):
  newCommand=[]
  for word in command.split(" "):
    try:
      if word[0:2] == "${":
        end=word.find("}")
        newCommand.append(os.environ[word[2:end]])
      elif word[0] == "$":
        newCommand.append(os.environ[word[1:]])
      else:
        newCommand.append(word)
    except KeyError:
      print "Cannot find environment variable %s." % (word)
      newCommand.append(word)
  return(" ".join(newCommand))

# Install handler for updating "status" file on premature termination
if os.name == "nt":
  try:
    import win32api
    win32api.SetConsoleCtrlHandler(handleSignals, True)
  except ImportError:
    raise Exception("Cannot import pywin32 module.")
else:
  import signal
  signal.signal(signal.SIGTERM, handleSignals)
# If we run on Windows, use the batch relay engines scripts
# for lammps (1 CPU) or mpiexec ( n CPUS)
# if we run on posix systems, use the native scripts
ext=".bat"
filePrefix="pdms_lammps"
BatchfilePrefix=filePrefix.replace("&","MAPSAMP")
CREATE_NO_WINDOW = 0x08000000
if os.name == "posix" :
   ext=""
   BatchfilePrefix=filePrefix
   CREATE_NO_WINDOW = 0x00000000
# Change to the directory this script is located in
path=os.path.dirname(sys.argv[0])
if len(path) != 0:
  os.chdir(path)

writeCommandLog("# log for run.py ","w")
writeStatus("Running " + str(os.getpid()))
if len(sys.argv) == 1:
  nCPU=1
  useGPU=0
elif len(sys.argv) == 2:
  if sys.argv[1] == "-g":
    nCPU=1
    useGPU=1
  else:
    nCPU=int(sys.argv[1])
    useGPU=0
else:
  nCPU=int(sys.argv[2])
  useGPU=1

(prefarch, prefmode) = getLammpsPreferences()

if os.name != "posix" and nCPU > 1 and useGPU > 0:
  if prefmode == "GPU":
    writeCommandLog("warning: disable multiple CPUs with GPU on windows")
    nCPU = 1
  elif prefmode == "CPUGPU":
    writeCommandLog("warning: using GPU with multiple CPUs on windows")
lammpsExe="lammps"
cuda=""
omp=""
if useGPU == 1:
  (capability, cuda, omp)=checkGPUCapabilities()
  if not capability:
    writeCommandLog("warning: will not use GPU"+omp)
    useGPU=0
  cudaVersion=0
  if cuda!="":
    cudaVersion=cuda.split(":")[1].strip().replace(".","")
  if int(cudaVersion) < 35:
    if omp == "":
      writeCommandLog("warning: will not use GPU"+omp)
      useGPU = 0
    else:
      if os.name != "posix":
        writeCommandLog("warning: disable multiple CPUs with GPU_OPENCL " + omp)
        nCPU = 1
supportGPU = True
if useGPU == 1:
  supportGPU = setGPUKeywords("pdms_lammps.in")
  if lammpsExe == "lammps" and supportGPU:
    lammpsExe = getExecutableName(cuda, omp, prefarch)
else:
  removeGPUKeywords("pdms_lammps.in")

if not supportGPU:
  useGPU = 0
  removeGPUKeywords("pdms_lammps.in")

if os.name == "posix":
  lammps=findInPath(lammpsExe)
  if lammps == None:
    msg = "Could not find lammps executable ("+lammpsExe+") in the path."
    print msg
    writeCommandLog("error : "+msg)
    writeStatus("Failed")
    sys.exit(1)

writeCommandLog("executable : "+lammpsExe)
mpiexec=checkEnvironment("mpiexec"+ext)
command=mpiexec.split(" ")
for parameter in ["-np", str(nCPU), lammpsExe, "-in", BatchfilePrefix + ".in", "-log", BatchfilePrefix + ".log", "-screen", BatchfilePrefix + ".err"]:
  command.append(parameter)
if os.name != "posix" and nCPU > 1 and useGPU > 0:
  if not "-localonly" in command:
    writeCommandLog("info : -localonly option was added to mpiexec")
    command.insert(1, "-localonly")
writeCommandLog("command : " + " ".join(command))
exitValue=subprocess.call(command,creationflags=CREATE_NO_WINDOW)
if exitValue != 0:
  writeStatus("Failed")
else:
  writeStatus("Completed")
