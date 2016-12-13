# Chatbot Training Automation
* This is a repository of all the codes and scripts for chatbot training automation.
* Actually, [the repository for chatbot training client/server scripts](https://github.com/aihoon/Chatbot_Training_Scripts) was released in 16/12/05. <br/>
  However, it was found that all the Python codes and shell scripts for chatbot training had **NOT** been managed and  
  even didn't work.
* The directory structure was re-built, all the Python codes and shell scripts was re-visited, and the chatbot training automation flow was organized.
* From the viewpoint of user, the chatbot training flow is managed by only one shell script, <br/>
  ***chatbot_training_client.sh*** for Linux and ***chatbot_training_client.bat*** for WINDOWS.

# Prerequisite

## Windows OS
* Install OpenSSH 7.3p1-2 via https://www.mls-software.com/files/setupssh-7.3p1-2.exe <br/>
  It is noted that the other version of OpenSSH MUST be uninstalled.
* Install WinScp via https://winscp.net/dwonload/WinSCP-5.9.3-Setup.exe

## OS X
* None

## Ubuntu
* None

# Chatbot training procedure

1. Prepare ${PROJECT_NAME}.test.txt and {$PROJECT_NAME}.train.txt in any directory(${BIN}).

2. Open Console and go to ${BIN} directory.

3. Download client shell file in the ${BIN} directory. <br/>
   chatbot_training_client.bat for WINDOWS or chatbot_training_client.sh for Linux.

4. start the chatbot training with "start" command and test/training text data;  <br/>
   *BASH/DOS> chatbot_training_client.{sh,bat} start ${LANGUAGE} ${PROJECT_NAME} ${SERVER_NUM}* <br/>
   where ${SERVER_NUM} is the last IP of the server based on 10.122.64.\* <br/>
   and ${LANGUAGE} is kor or eng.
   
5. The training progress can be checked with "check" command; <br/>
   *BASH/DOS> chatbot_training_client.{sh,bat} check ${LANGUAGE} ${PROJECT_NAME} ${SERVER_NUM}* <br/>

6. The training process can be killed with "stop" command; <br/>
   *BASH/DOS> chatbot_training_client.{sh,bat} stop ${LANGUAGE} ${PROJECT_NAME} ${SERVER_NUM}* <br/>

7. The HTTP server with the trained chatbot can be run; <br/>
   *BASH/DOS> chatbot_training_client.{sh,bat} run_server ${LANGUAGE} ${PROJECT_NAME} ${SERVER_NUM}* <br/>
   It is noted that if the HTTP server with the same trained chatbot exists, it should be automatically killed and re-runed with new trained chatbot and this HTTP server should be automatically killed in one day.
   
7. The HTTP server with the trained chatbot can be killed; <br/>
   *BASH/DOS> chatbot_training_client.{sh,bat} kill_server ${LANGUAGE} ${PROJECT_NAME} ${SERVER_NUM}* <br/>

# Remark
* The chatbot training automation package with the Python codes and shell scripts was installed only in 10.122.64.63. <br/>
  If you want to train the chatbot in the other server, just let aihoon know the server IP.
  
