#!/bin/bash

if [ $# -ne 1 ]
then
	echo Usage: $0: "ProjectName"
	exit
fi

PROJ="$1"
LANG="eng"

BASE_DIR="${HOME}/Mindsbot"
SCR_DIR="${BASE_DIR}/Script"
TAR_DIR="${BASE_DIR}/Train.${LANG}/${PROJ}"

PREPROCESSOR__PY="${SCR_DIR}/preprocess_eng.py"
MK_SUBTITLE_DATA__PY="${SCR_DIR}/mk_subtitle_data.py"
TEXT2WFREQ="${SCR_DIR}/text2wfreq"

train_file=${PROJ}.train.txt
test_file=${PROJ}.test.txt

echo -e " # Preprocessing - training file, $train_file"
python ${PREPROCESSOR__PY} -d ${TAR_DIR}/${train_file} > /dev/null

echo -e " # Preprocessing - test file, $test_file"
python ${PREPROCESSOR__PY} -d ${TAR_DIR}/${test_file} > /dev/null

echo -e " # Preprocessing - vocab.txt"
# we convert upper case to lower case
cat ${TAR_DIR}/${PROJ}.t*.txt | tr ['A-Z'] ['a-z'] | ${TEXT2WFREQ} | sort -n -r -k 2 > /dev/null > ${TAR_DIR}/${PROJ}.vocab.freq
echo 'UNK 0'  >  ${TAR_DIR}/${PROJ}.vocab.txt
echo '</s> 0' >>  ${TAR_DIR}/${PROJ}.vocab.txt
# assume that vocabulary size is 50000. --> 49998 + 2 (UNK, </s>)
cat ${TAR_DIR}/${PROJ}.vocab.freq | grep -v '</s>' | head -n 49998 > /dev/null >> ${TAR_DIR}/${PROJ}.vocab.txt

echo -e " # Preprocessing - training"
python ${MK_SUBTITLE_DATA__PY} -v ${TAR_DIR}/${PROJ}.vocab.txt \
                               -s ${TAR_DIR}/${PROJ}.train.txt.prev.txt \
                               -t ${TAR_DIR}/${PROJ}.train.txt.next.txt \
                               -o ${TAR_DIR}/${PROJ}.train.pkl > /dev/null

echo -e " # Preprocessing - test"
python ${MK_SUBTITLE_DATA__PY} -v ${TAR_DIR}/${PROJ}.vocab.txt \
                               -s ${TAR_DIR}/${PROJ}.test.txt.prev.txt \
                               -t ${TAR_DIR}/${PROJ}.test.txt.next.txt \
                               -o ${TAR_DIR}/${PROJ}.test.pkl > /dev/null

gzip ${TAR_DIR}/*.pkl

