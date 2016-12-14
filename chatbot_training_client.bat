@echo off

set "BASE_DIR=/home/msl/Mindsbot"
set "SCR_DIR=%BASE_DIR%/Script"
set "BASE_IP=10.122.64"
set "CHATBOT_SERVER__SH=chatbot_training_server.sh"
set "CHATBOT_CLIENT__SH=chatbot_training_client.sh"
set "SERVER_PEM_FILE=/home/msl/.ssh/id_rsa"
set "HTTP_URL__TXT=http_url.txt"

::########################################################################
:: Check input arguments...


IF "%4"=="" (
    echo.
    echo.  # USAGE
    echo.    $ chatbot_training_client.bat COMMAND LANGUAGE PROJECT_NAME SERVER_NUMBER
    echo.      where COMMAND is start, check, stop, remove, run_server, or kill_server.
    echo       LANG is kor or eng.
    echo.
    goto :eof
)

::########################################################################
:: Define variables...

set "CMD_TYPE=%1"
set "LANG=%2"
set "PROJ_NAME=%3"
set "SERVER_NUM=%4"

set "TEST_FILE=%PROJ_NAME%.test.txt"
set "TRAIN_FILE=%PROJ_NAME%.train.txt"
set "PEM_FILE=msl_%SERVER_NUM%.pem"
set "SERVER_IP=10.122.64.%SERVER_NUM%"

::########################################################################
:: Main.

CALL :SUB_CHECK_PEM

IF "%CMD_TYPE%"=="start" (

    echo.
    echo. # Check if test file exists...
    IF NOT EXIST %TEST_FILE% (
        echo. @ Error: file not found, %TEST_FILE%
        echo.
        goto :eof
    )

    echo.
    echo. # Check if test file exists...
    IF NOT EXIST %TRAIN_FILE% (
        echo. @ Error: file not found, %TRAIN_FILE%
        echo.
        goto :eof
    )

    echo.
    echo. # Copy test and train files to server.
    scp -i %PEM_FILE% %PROJ_NAME%.*.txt msl@%SERVER_IP%:%BASE_DIR%
)


IF "%CMD_TYPE%"=="start"       goto :cond
IF "%CMD_TYPE%"=="check"       goto :cond:
IF "%CMD_TYPE%"=="stop"        goto :cond
IF "%CMD_TYPE%"=="remove"      goto :cond
IF "%CMD_TYPE%"=="run_server"  goto :cond
IF "%CMD_TYPE%"=="kill_server" goto :cond

echo.
echo. @ Error: command not defined, %CMD_THYPE%.
echo.          Try again with start, check, stop, run_server, or kill_server.
echo.
goto :eof

:cond
    
    set SERVER_CMD="cd %SCR_DIR%; ./%CHATBOT_SERVER__SH% %CMD_TYPE% %LANG% %PROJ_NAME%"
    ssh -i %PEM_FILE% msl@%SERVER_IP% %SERVER_CMD%
        
        IF "%CMD_TYPE%"=="run_server" (
        del /s %HTTP_URL__TXT%
        scp -i %PEM_FILE% msl@%SERVER_IP%:%BASE_DIR%/%HTTP_URL__TXT% .
        set /p HTTP_URL=<%HTTP_URL__TXT%
                echo. %HTTP_URL% Start...
                start %HTTP_URL%
        )
    goto :eof

:eof


::########################################################################
:: Subroutines

:SUB_CHECK_PEM
    
    IF NOT EXIST %PEM_FILE% (

        echo.
        echo. # There is no security certificate. Try to find it.
        echo.
        scp msl@%SERVER_IP%:%SERVER_PEM_FILE% %PEM_FILE%
        chmod 0400 %PEM_FILE%

    )
    
    EXIT /B
