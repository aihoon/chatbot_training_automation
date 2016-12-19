#!/bin/bash

########################################################################

if [ "$#" -ne "3" ]; then
    echo -e "\n USAGE: chatbot_training_server.sh COMMAND LANG PROJECT_NAME "
    echo -e "        where COMMAND is start, check, stop, or remove, run_server, or stop_server"
    echo -e "              LANG is kor or eng.\n"
    exit
fi

########################################################################

CMD="$1"
LANG="$2"
PROJ_NAME="$3"

if [ "${LANG}" != "kor" ] && [ "${LANG}" != "eng" ]; then
    echo -e "\n @ Error: LANG(\"${LANG}\"), MUST be \"kor\" or \"eng\"\n"
    exit
fi

TEST_FILE=${PROJ_NAME}.test.txt
TRAIN_FILE=${PROJ_NAME}.train.txt

SRC_DIR="${HOME}/Mindsbot"
TAR_DIR="${HOME}/Mindsbot/Train.${LANG}/${PROJ_NAME}"
SCR_DIR="${HOME}/Mindsbot/Script"
export PATH=${SCR_DIR}:${PATH}
CHECK_FILE="check.txt"

MK_DATA__SH="mk_data_${LANG}.sh"
TRAIN__PY="${SCR_DIR}/train.py"
SERVER_HTML__PY="${SCR_DIR}/server_html.py"

########################################################################

run_server () {

    echo -e " # HTTP server is starting...\n"
    cd ${TAR_DIR}
    python ${SERVER_HTML__PY} ${PROJ_NAME} ${LANG} ${MY_IP} ${MY_PORT} > ${TAR_DIR}/${PROJ_NAME}_server.log 2>&1 &
    (sleep 24h; stop_server) &
    
    echo -e "-------------------------------------------------------------------"
    echo -e " # Now, you can check the chatbot training result in a few minutes"
    echo -e "   from http://${MY_IP}:${MY_PORT}/?query=YOUR_QUESTION"
    echo -e "   It is noted that this HTTP server will be killed in a day."
    echo -e "-------------------------------------------------------------------\n"

}

stop_server () {

    PID=$(ps aux | grep -e "python.${SERVER_HTML__PY}.*${PROJ_NAME}" | grep -v "grep" | awk '{print $2;}');
    if [ ! -z ${PID} ]; then
        echo -e " # HTTP server is killed..."
        kill -9 ${PID}
    fi 
}

find_empty_port () {

    BASE_PORT="9500"

    for idx in {9500..9510}
    do
        USED=$(lsof -i :${idx})
        if [ -z ${USED} ]; then
            MY_PORT="${idx}"
            echo -e " # ${idx} port is selected for your HTTP application."
            break
        fi
        echo -e " # ${idx} port is being used."
    done

}


########################################################################
echo

if [ "${CMD}" == "start" ]; then
    
    echo -e " # Check if ${TEST_FILE} exists."
    if [ ! -f "${SRC_DIR}/${TEST_FILE}" ]; then
        echo -e "\n @ Error: test file not found ${TEST_FILE}\n"
        exit
    fi
    
    echo -e " # Check if ${TRAIN_FILE} exists."
    if [ ! -f ${SRC_DIR}/${TRAIN_FILE} ]; then
        echo -e " \n@ Error: train file not found ${TRAIN_FILE}\n"
        exit
    fi
    
    echo -e " # Check if ${TAR_DIR} exists."
    if [ -d ${TAR_DIR} ]; then
        echo -e "\n @ Error: ${TAR_DIR} directory already exists.\n"
        exit
    fi
    
    echo -e "\n # Make project directory, ${PROJ_NAME} and copy the test and train files to it."
    mkdir ${TAR_DIR}
    mv -f ${SRC_DIR}/${TEST_FILE}  ${TAR_DIR}
    mv -f ${SRC_DIR}/${TRAIN_FILE} ${TAR_DIR}
    
    echo -e "\n # Run preprocessing, ${MK_DATA__SH}.\n"
    cd ${TAR_DIR}
    ${MK_DATA__SH} ${PROJ_NAME}
    
    echo -e "\n # Run training ...\n"
    THEANO_FLAGS=device=gpu nohup python ${TRAIN__PY} ${PROJ_NAME} ${LANG} > ${PROJ_NAME}.log 2>&1 &
    echo " # Training process"
    echo -n " > " 
    ps -efl | grep "python.*${PROJ_NAME}" | grep -v grep
    echo
    echo -e " ####################################"
    echo -e " # Check the result in a few hours..."
    echo -e " ####################################"
    
#-----------------------------------------------------------------------

elif [ "${CMD}" == "check" ]; then

    ps aux | grep -e python | grep -e ${PROJ_NAME} | grep -v grep > ${CHECK_FILE}

    if [ ! -s ${CHECK_FILE} ]; then
        echo -e " # Training process for ${PROJ_NAME} doesn't exist. Training might be done."
    else
        echo -e " # Training process for ${PROJ_NAME} is still running..."
        echo -e "\n # The followings are the current logging information..."
        echo -e "---------------------------------------------------------"
        timeout 3 tail -f ${TAR_DIR}/${PROJ_NAME}.log
        sleep 2
        echo -e "---------------------------------------------------------"
    fi

#-----------------------------------------------------------------------
    
elif [ "${CMD}" == "stop" ]; then

    PID=$(ps aux | grep -E "python.*${PROJ_NAME}" | grep -v grep | awk '{print $2;}');
    if [ -z ${PID} ]; then
        echo -e " # Training process for ${PROJ_NAME} doesn't exist. Training might be done."
    else
        echo -e " # Training process for ${PROJ_NAME} is going to be killed."
        kill -9 ${PID}
    fi

#-----------------------------------------------------------------------

elif [ "${CMD}" == "remove" ]; then

    PID=$(ps aux | grep -E "python.*${PROJ_NAME}" | grep -v grep | awk '{print $2;}');
    if [ -z ${PID} ]; then
        echo -e " # Training process for ${PROJ_NAME} doesn't exist."
    else
        echo -e " # Training process for ${PROJ_NAME} is killed."
        kill -9 ${PID}
    fi

    if [ -d ${TAR_DIR} ]; then
        echo -e " # Training directory, ${TAR_DIR} is deleted."
        rm -rf ${TAR_DIR}
    else
        echo -e " # Training directory, ${TAR_DIR}, doesn't exist."
    fi

#-----------------------------------------------------------------------

elif [ "${CMD}" == "run_server" ]; then

    PID=$(ps aux | grep -e "python.${SERVER_HTML__PY}.*${PROJ_NAME}" | grep -v "grep" | awk '{print $2;}');

    if [ ! -z ${PID} ]; then
        echo -e "\n @ Warning: HTTP server for ${PROJ_NAME} is still running."
        echo -e "   \"stop_server\" MUST be called before \"run_server\"...\n"  
    else

        MY_IP=$(ip addr | grep 'state UP' -A2 | tail -n1 | awk '{print $2}' | cut -f1  -d'/')
        MY_PORT=0
        find_empty_port
        run_server 

    fi

#-----------------------------------------------------------------------

elif [ "${CMD}" == "kill_server" ]; then

    PID=$(ps aux | grep -e "python.${SERVER_HTML__PY}.*${PROJ_NAME}" | grep -v "grep" | awk '{print $2;}');

    if [ ! -z ${PID} ]; then
        echo -e " # HTTP server of ${PROJ_NAME} is killed..."
        kill -9 ${PID}
    else
        echo -e " # HTTP server of ${PROJ_NAME} does not exist."
    fi

#-----------------------------------------------------------------------


else
    
    echo -e "\n @ Error: command not found, ${CMD}.\n"
    exit

fi

echo

