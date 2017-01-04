#!/bin/bash

REMOTE_CONTROL="NO"
MSL63="125.132.250.204" 

BASE_DIR="/home/msl/Mindsbot"
SCR_DIR="${BASE_DIR}/Script"
BASE_IP="10.122.64"
CHATBOT_CLIENT__SH="chatbot_training_client.sh"
CHATBOT_SERVER__SH="chatbot_training_server.sh"
SERVER_PEM_FILE="/home/msl/.ssh/id_rsa"

###########################################################################
#   SUBROUTINES...      

_check_pem () {

    if [ ! -f ${PEM_FILE} ]; then

        echo -e "\n # There is no security certificate. Try to find it."
        scp msl@${SERVER_IP}:${SERVER_PEM_FILE} ${PEM_FILE}
        chmod 0400 ${PEM_FILE} 

    fi
}

###########################################################################
#   Check input arguments...

if [ "$#" -ne "4" ] 
then
    echo 
    echo " # USAGE"
    echo " \$ ${CHATBOT_CLIENT__SH} LANG PROJECT_NAME SERVER_NUMBER COMMAND"
    echo "   where COMMAND is start, check, stop, remove, run_server, check_server, kill_server, or show_servers."
    echo "             LANG is kor or eng."
    exit
fi

###########################################################################
# Define variables...

LANG="$1"
PROJ_NAME="$2"
SERVER_NUM="$3"
CMD_TYPE="$4"

TEST_FILE=${PROJ_NAME}.test.txt
TRAIN_FILE=${PROJ_NAME}.train.txt
PEM_FILE=msl_${SERVER_NUM}.pem
URL_FILE=${PROJ_NAME}.url
if [ ${REMOTE_CONTROL} == "YES" ]; then
    SERVER_IP=${MSL63}
else
    SERVER_IP=${BASE_IP}.${SERVER_NUM}
fi


###########################################################################
# MAIN.

_check_pem

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
    scp -i ${PEM_FILE} ${PROJ_NAME}.*.txt  msl@${SERVER_IP}:${BASE_DIR}

    echo -e "\n # Run the server training script, \"${CHATBOT_SERVER__SH}\".\n"

fi

if [ "${CMD_TYPE}" == "start"        ] || 
   [ "${CMD_TYPE}" == "check"        ] || 
   [ "${CMD_TYPE}" == "stop"         ] || 
   [ "${CMD_TYPE}" == "remove"       ] || 
   [ "${CMD_TYPE}" == "run_server"   ] ||
   [ "${CMD_TYPE}" == "check_server" ] ||
   [ "${CMD_TYPE}" == "kill_server"  ] ||
   [ "${CMD_TYPE}" == "show_servers" ]; then

    SERVER_CMD="cd ${SCR_DIR}; ./${CHATBOT_SERVER__SH} ${CMD_TYPE} ${LANG} ${PROJ_NAME}"
    ssh -i ${PEM_FILE} msl@${SERVER_IP} ${SERVER_CMD}

else

    echo -e "\n @ Error: command, \"${CMD_TYPE}\", is NOT defined.\n"

fi

if [ "${CMD_TYPE}" == "run_server"  ]; then

    rm -f ${URL_FILE}
    SERVER_CMD="[[ -f ${BASE_DIR}/${URL_FILE} ]] && echo YES || echo NO;"
    ANS=$(ssh -i ${PEM_FILE} msl@${MSL63} ${SERVER_CMD})
    if [ ${ANS} == "YES" ]; then
        scp -i ${PEM_FILE} msl@${SERVER_IP}:${BASE_DIR}/${URL_FILE} .
        ssh -i ${PEM_FILE} msl@${SERVER_IP} rm -f ${BASE_DIR}/${URL_FILE}

        if [ ${REMOTE_CONTROL} == "YES" ]; then
            HTTP_URL=$(sed "s/${BASE_IP}.${SERVER_NUM}/${SERVER_IP}/g" ${URL_FILE})
        else
            HTTP_URL=$(cat ${URL_FILE}) 
        fi

        echo -e " # Open the webpage for ${PROJ_NAME} test..."
        python -m webbrowser "${HTTP_URL}"
    else
        echo " @ URL file not found."
    fi

fi
