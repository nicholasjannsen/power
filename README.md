# Timeseries_Tools
This repository is library of tools used in Time Series Analysis in  astronomy. It is developed toward science in Asteroseismology and Exoplanets. 

# DOCUMENTATIONS OF ANDOR SLIT-VIEWER SEQUENCER COMMANDS

The documentation presented here is linked to the task to *Map pyandor commands to NOT sequencer commands*. All code related to this subject is written in Python and is used as a wrapper around the c interface of the Andor camera API commands that internally controls the camera software. The following is an overview of the specific project tasks:    
  - Develop sequencer commands of identically functionallity as current systems operate
  - Handle parsing arguments over the command line and occations of potential flags
  - Create a stable TCP connections between a 'local' server and the 'nte1' server
  - Create a 'client' program (on the local server) that handles all communication

The main communicational structure of these task can be described as following:
  1. Execution of sequencer command (e.g., `xbin 2 -d`)
  2. Parameters and flags are parsed to **client**
  3. **client** etablish tcp connection to **nte1-server**
  4. **nte1-server** calls offending Andor API command for camera response
  5. API error code is returned to the **nte1-server** and send back to **client**
  6. **client** identify the exact message beloging to the error code
  7. Message is sent to the Talker as communicational link to the user interface
If the debug option is activated, all messages will be reported back to the Talker.  

Sequencer commands:
--- 
Sequencer commands are written in python but do not have any extensions such as '.py'. All commands are developed using the [STANCAM SEQUENCER Command Reference](http://www.not.iac.es/observing/seq/stancam-seq-commands.html) and the information stated within the "Andor_Software_Development_Kit_2009.pdf". Each sequencer command is internally linked to the 'client' program where parsed arguments are posible using the Python `optget` module, whose API is designed to be familiar to users of the C `getopt()` function. A usage/help function activated by the flag `-h` is available from the command line, and too many parsing parameters will results in an error and a print of the usage function.  As an requirement this function also include the debug option activated by the flag `-d`. Each sequencer command has been tailored for its specific use and succes in parsing arguments to the 'client' program.   

Client:
---
As mentioned above the *client* program for for the Andor slit-viewer camera. This program takes care of error code handling, argument parsing, communication back and forth to the *nte1-server*, and writing to the (syslog) Talker. This program is written in Python3, as this will keep it stable and adaptable to future updates. Notice that the `socket` option with the `with` statement, used to nicely clean the tcp connection, is only supported in Python3 and, hence, this will fail if using Python2. The *client* creates a stable tcp connection to the *nte1-server* and parse up to 2 arguments needed to execute the corresponding API commands. All information parsed over the tcp connection are encoded in bytes. When the API error code, specified by a 5 digit number, is returned to the *client*, the *client* will use the corresponding sequecer command name and API function to search for the proper error code string (e.g. "DRV\_SUCCESS"), API description ("All parameters accepted"), and error message (e.g. "[ASTRO], [NOTE], [ERROR], or [WARNING]"). An example in a sucessful use of the `xbin 2` command results in the Talker message: `[NOTE]: DRV\_SUCCESS: All parameters accepted`.      

Server nte1@obs:
---
The *server* takes care of the communication back and forth to the *client* program, and the return message from the client. This piece of software is to be placed at the nte1 server and therefore needs to be compatible with the Python 2.3. for which is used for the Andor camera. The server unpacks potential sequencer command arguments and call the corresponding API commands, for which all are available as a Python wrapper function within the *server*. It is assumed that the *server* knows the exact Andor camera state to all times, and if the camera is powered off, the camera startup/initializatio procedure will fetched by the *server* upon a call to the `NTE_pyandor.py` wrapper. This wrapper has been modified in this work some store the default image parameters and to secure a **Safty shutdown procedure**.  

Talker:
---
Main program for communicating with the Talker using the Python `syslog` interface. This function takes the called sequencer ("seqcom": string) as input as well as the message ("message": string) needed to report to the Talker and prompt back when the Talker communicate back its writting. 

Future prospects:
---
  - Test if *server* is compatible with Python 2.3
  - Test if `NTE_pyandor.py` can store image settings for the *server* to fetch
  - Test if communication to the Talker works 
  - It seems like that while using the same tcp port for all communication, the port is sometimes blocked for no obvious reason, hence this needs to be resolved, as we want to communicate continously.
  - The exposure sequencer commands (`exp/expose`, `mexp/mexpose`, `dark`, and `mdark`) are the most complecated in this process and rely on several API request/settings. Since only harcode values within these API function are used to communicate with the Andor camera, these should in principle not fail, and the returned error code will be allocated to the `SetExposureTime` API function. Hence, if some of the previous commands fails for some reason, it will not be immediately obvious which API is was and the exposure might run with dubious settings.
