#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import time
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

def getGromacsPreferences():
  arch=""
  mode=""
  preffilename = "gromacs_pref"
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
# for gromacs (1 CPU) or mpiexec ( n CPUS)
# if we run on posix systems, use the native scripts
ext=".bat"
CREATE_NO_WINDOW = 0x08000000
if os.name == "posix" :
   ext=""
   CREATE_NO_WINDOW = 0x00000000
# Change to the directory this script is located in
path=os.path.dirname(sys.argv[0])
if len(path) != 0:
  os.chdir(path)

writeCommandLog("# log for run.py ","w")
writeStatus("Running " + str(os.getpid()))
filePrefix="pdes8200"
if os.name == "posix" :
   BatchfilePrefix=filePrefix
else:
   BatchfilePrefix=filePrefix.replace("&","MAPSAMP")
BatchfileTpr=BatchfilePrefix+".tpr"
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

(prefarch, prefmode) = getGromacsPreferences()

# Get the path to MPI executable
gromacsExe="gmx_mpi"
if os.name == "posix":
  gromacs=findInPath(gromacsExe)
  if gromacs == None:
    msg = "Could not find gromacs executable ("+gromacsExe+") in the path."
    print msg
    writeCommandLog("error : "+msg)
    writeStatus("Failed")
    sys.exit(1)

# Get the path to GPU executable
if useGPU == 1 and os.name == "nt":
  gromacsGPUExe="gmx_cuda"
if os.name == "posix":
  if useGPU == 1:
    writeCommandLog("executable : "+gromacsExe+" with GPU support" )
  else:
    writeCommandLog("executable : "+gromacsExe)
else:
  if useGPU == 1:
    writeCommandLog("executable : "+gromacsGPUExe)
  else:
    writeCommandLog("executable : "+gromacsExe)
mpiexec=checkEnvironment("mpiexec"+ext)
if nCPU == 1 and useGPU == 0:
  command=[gromacsExe+ext, "mdrun", "-cpo", "-v","-nb",  "cpu", "-ntomp",  "1", "-deffnm", BatchfilePrefix]
  errFile=open(os.devnull,"w")
elif nCPU == 1 and useGPU == 1:
  if os.name == "posix":
    command=[gromacsExe+ext, "mdrun", "-cpo", "-v","-nb",  "gpu", "-ntomp",  "1", "-deffnm", BatchfilePrefix]
    errFile=open(os.devnull,"w")
  else:
    command=[gromacsGPUExe+ext, "mdrun", "-cpo", "-v","-nb",  "gpu", "-ntomp",  "1", "-deffnm", BatchfilePrefix]
    errFile=open(os.devnull,"w")
elif nCPU > 1 and useGPU == 1:
  command=mpiexec.split(" ")
  if os.name == "posix":
    for parameter in ["-np", str(nCPU), gromacsExe, "mdrun", "-cpo", "-v","-nb",  "gpu",  "-ntomp",  "1", "-deffnm", BatchfilePrefix]:
      command.append(parameter)
    errFile=open(os.devnull,"w")
  else:
    for parameter in ["-np", str(nCPU), gromacsGPUExe, "mdrun", "-cpo", "-v","-nb",  "gpu",  "-ntomp",  "1", "-deffnm", BatchfilePrefix]:
      command.append(parameter)
    errFile=open(os.devnull,"w")
elif nCPU > 1 and useGPU == 0:
  command=mpiexec.split(" ")
  for parameter in ["-np", str(nCPU), gromacsExe, "mdrun", "-cpo", "-v","-nb", "cpu",  "-ntomp",  "1", "-deffnm", BatchfilePrefix]:
    command.append(parameter)
  errFile=open(os.devnull,"w")
writeCommandLog("command : " + " ".join(command))
exitValue=subprocess.Popen(command,stderr=errFile,creationflags=CREATE_NO_WINDOW)
time.sleep(50)
fcheck=open("pdes8200.log").read()
try:
	i=fcheck.index("Fatal error:")
	if (i>0):
		exitValue.kill()
except:
	exitValue.communicate()
errFile.close()
exitValue = exitValue.returncode
if exitValue != 0:
  writeStatus("Failed")
else:
  writeStatus("Completed")
