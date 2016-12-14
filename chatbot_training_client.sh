#!/bin/bash

BASE_DIR="/home/msl/Mindsbot"
SCR_DIR="${BASE_DIR}/Script"
BASE_IP="10.122.64"
CHATBOT_CLIENT__SH="chatbot_training_client.sh"
CHATBOT_SERVER__SH="chatbot_training_server.sh"
SERVER_PEM_FILE="/home/msl/.ssh/id_rsa"
HTTP_URL__TXT="http_url.txt"

###########################################################################
#   SUBROUTINES...      

check_pem () {

    if [ ! -f ${PEM_FILE} ]; then

        echo -e "\n # There is no security certificate. Try to find it."
        scp msl@${SERVER_IP}:${SERVER_PEM_FILE} ${PEM_FILE}
        chmod 0400 ${PEM_FILE} 

    else

        chmod 0400 ${PEM_FILE} 

    fi
}

###########################################################################
#   Check input arguments...

if [ "$#" -ne "4" ] 
then
    echo 
    echo " # USAGE"
    echo " \$ ${CHATBOT_CLIENT__SH} COMMAND LANG PROJECT_NAME SERVER_NUMBER"
    echo "   where COMMAND is start, check, stop, remove, run_server, or kill_server."
    echo "             LANG is kor or eng."
    exit
fi

###########################################################################
# Define variables...

CMD_TYPE="$1"
LANG="$2"
PROJ_NAME="$3"
SERVER_NUM="$4"

TEST_FILE=${PROJ_NAME}.test.txt
TRAIN_FILE=${PROJ_NAME}.train.txt
PEM_FILE=msl_${SERVER_NUM}.pem
SERVER_IP=${BASE_IP}.${SERVER_NUM}

###########################################################################
# MAIN.

check_pem

if [ "${CMD_TYPE}" == "start" ]; then

    # Check if test file exists...
    if [ ! -f ${TEST_FILE} ]; then
        echo -e "\n @ Error: file not found, ${TEST_FILE}\n"
        exit
    fi

    # Check if train file exists...
    if [ ! -f ${TRAIN_FILE} ]; then
        echo -e "\n @ Error: file not found, ${TRAIN_FILE}.\n"
        exit
    fi

    echo -e "\n # Copy test and train files to server.\n"
    scp -i ${PEM_FILE} ${TEST_FILE}  msl@${SERVER_IP}:${BASE_DIR}
    scp -i ${PEM_FILE} ${TRAIN_FILE} msl@${SERVER_IP}:${BASE_DIR}

    echo -e "\n # Run the server training script, \"${CHATBOT_SERVER__SH}\".\n"

fi

if [ "${CMD_TYPE}" == "start"       ] || 
   [ "${CMD_TYPE}" == "check"       ] || 
   [ "${CMD_TYPE}" == "stop"        ] || 
   [ "${CMD_TYPE}" == "remove"      ] || 
   [ "${CMD_TYPE}" == "run_server"  ] ||
   [ "${CMD_TYPE}" == "kill_server" ]; then

    SERVER_CMD="cd ${SCR_DIR}; ./${CHATBOT_SERVER__SH} ${CMD_TYPE} ${LANG} ${PROJ_NAME}"
    ssh -i ${PEM_FILE} msl@${SERVER_IP} ${SERVER_CMD}

else

    echo -e "\n @ Error: command, \"${CMD_TYPE}\", is NOT defined.\n"

fi

if [ "${CMD_TYPE}" == "run_server"  ]; then

    scp -i ${PEM_FILE} msl@${SERVER_IP}:${BASE_DIR}/${HTTP_URL__TXT} .
    HTTP_URL=$(cat ${HTTP_URL__TXT}) 
    python -m webbrowser "${HTTP_URL}"

fi
