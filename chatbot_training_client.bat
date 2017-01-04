@echo off

setlocal enabledelayedexpansion

set "REMOTE_CONTROL=NO"
set "MSL63=125.132.250.204"
set "TEMP_FILE=temp.txt"

set "BASE_DIR=/home/msl/Mindsbot"
set "SCR_DIR=%BASE_DIR%/Script"
set "BASE_IP=10.122.64"
set "CHATBOT_SERVER__SH=chatbot_training_server.sh"
set "CHATBOT_CLIENT__SH=chatbot_training_client.sh"
set "SERVER_PEM_FILE=/home/msl/.ssh/id_rsa"

::########################################################################
:: Check input arguments...


IF "%4"=="" (
    echo.
    echo.  # USAGE
    echo.    $ chatbot_training_client.bat LANGUAGE PROJECT_NAME SERVER_NUMBER COMMAND
    echo.      where COMMAND is start, check, stop, remove, run_server, check_server, kill_server or show_servers.
    echo       LANG is kor or eng.
    echo.
    goto :eof
)

::########################################################################
:: Define variables...

set "LANG=%1"
set "PROJ_NAME=%2"
set "SERVER_NUM=%3"
set "CMD_TYPE=%4"

set "TEST_FILE=%PROJ_NAME%.test.txt"
set "TRAIN_FILE=%PROJ_NAME%.train.txt"
set "PEM_FILE=msl_%SERVER_NUM%.pem"
set "URL_FILE=%PROJ_NAME%.url"
IF "%REMOTE_CONTROL%"=="YES" (
    set "SERVER_IP=%MSL63%"
) ELSE (
    set "SERVER_IP=%BASE_IP%.%SERVER_NUM%"
)

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
    echo. # Check if train file exists...
    IF NOT EXIST %TRAIN_FILE% (
        echo. @ Error: file not found, %TRAIN_FILE%
        echo.
        goto :eof
    )

    echo.
    echo. # Copy test and train files to server.
    scp -i %PEM_FILE% %PROJ_NAME%.*.txt msl@%SERVER_IP%:%BASE_DIR%

    echo.
    echo. # Run the server training script, "%CHATBOT_SERVER__SH%"
    echo.
)


IF "%CMD_TYPE%"=="start"        goto :cond
IF "%CMD_TYPE%"=="check"        goto :cond
IF "%CMD_TYPE%"=="stop"         goto :cond
IF "%CMD_TYPE%"=="remove"       goto :cond
IF "%CMD_TYPE%"=="run_server"   goto :cond
IF "%CMD_TYPE%"=="check_server" goto :cond
IF "%CMD_TYPE%"=="kill_server"  goto :cond
IF "%CMD_TYPE%"=="show_servers" goto :cond

echo.
echo. @ Error: command not defined, %CMD_THYPE%.
echo.
goto :eof

:cond
    
    set SERVER_CMD="cd %SCR_DIR%; ./%CHATBOT_SERVER__SH% %CMD_TYPE% %LANG% %PROJ_NAME%"
    set SERVER_CMD2="[[ -f %BASE_DIR%/%URL_FILE% ]] && printf YES || printf NO;"
    ssh -i %PEM_FILE% msl@%SERVER_IP% %SERVER_CMD%

    IF "%CMD_TYPE%"=="run_server" (

        echo.
        echo. # Check if test HTTP URL file exists or not in server...
        del /s %URL_FILE% >nul 2>&1
        :: echo SERVER_CMD2 = %SERVER_CMD2%
        ssh -i %PEM_FILE% msl@%SERVER_IP% %SERVER_CMD2% > %TEMP_FILE%
        set /p ANS=<%TEMP_FILE%
        :: echo ANS is !ANS!
        IF "!ANS!"=="YES" (
            echo.
            echo. # Secure copy test HTTP URL file from server to local...
            echo.
            scp -i %PEM_FILE% msl@%SERVER_IP%:%BASE_DIR%/%URL_FILE% .

            IF "%REMOTE_CONTROL%"=="YES" (
                sed "s/%BASE_IP%.%SERVER_NUM%/%SERVER_IP%/g" %URL_FILE% > %TEMP_FILE%
                set /p HTTP_URL=<%TEMP_FILE%
            ) ELSE (
                set /p HTTP_URL=<%URL_FILE%
            ) 

            echo.
            echo. # Open the webpage for %PROJ_NAME% test via %HTTP_URL% 
            start !HTTP_URL!
        ) ELSE (
          echo.
          echo @ URL file not found.
          echo.
        )
    )

goto :eof

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

:eof
