#!/bin/sh

if [ $# -ne 1 ]
then
	echo Usage : $0: "ProjectName"
	exit
fi

PROJ="$1"
LANG="kor"

BASE_DIR="${HOME}/Mindsbot"
SCR_DIR="${BASE_DIR}/Script"
TAR_DIR="${BASE_DIR}/Train.${LANG}/${PROJ}"

PREPROCESSOR__PY="${SCR_DIR}/preprocess_${LANG}.py"
WORD2CHAR__PY="${SCR_DIR}/word2char.py"
MK_SR_DATA__PY="${SCR_DIR}/mk_SR_data.py"
TEXT2WFREQ="text2wfreq"

train_file=${PROJ}.train.txt
test_file=${PROJ}.test.txt

echo -e " # Preprocessing - training file, $train_file"
python ${PREPROCESSOR__PY} -d ${TAR_DIR}/$train_file > /dev/null
python ${WORD2CHAR__PY} -d ${TAR_DIR}/${train_file}.src.txt > /dev/null > ${TAR_DIR}/${PROJ}.train.src.char
python ${WORD2CHAR__PY} -d ${TAR_DIR}/${train_file}.tgt.txt > /dev/null > ${TAR_DIR}/${PROJ}.train.tgt.char

echo -e " # Preprocessing - test file, $test_file"
python ${PREPROCESSOR__PY} -d ${TAR_DIR}/$test_file > /dev/null
python ${WORD2CHAR__PY}    -d ${TAR_DIR}/${test_file}.src.txt > /dev/null > ${TAR_DIR}/${PROJ}.test.src.char
python ${WORD2CHAR__PY}    -d ${TAR_DIR}/${test_file}.tgt.txt > /dev/null > ${TAR_DIR}/${PROJ}.test.tgt.char

echo -e " # Preprocessing - vocab_word.txt"
cat ${TAR_DIR}/${PROJ}.t*.*.char | ${TEXT2WFREQ} | sort -n -r -k 2 > /dev/null > ${TAR_DIR}/${PROJ}.vocab_word.freq
echo 'UNK 0'  >  ${TAR_DIR}/${PROJ}.vocab_word.txt
echo '</s> 0' >> ${TAR_DIR}/${PROJ}.vocab_word.txt
cat ${TAR_DIR}/${PROJ}.vocab_word.freq | grep -v '</s>' > /dev/null >> /${TAR_DIR}/${PROJ}.vocab_word.txt

echo -e " # Preprocessing - training"
python ${MK_SR_DATA__PY} -v ${TAR_DIR}/${PROJ}.vocab_word.txt \
                         -s ${TAR_DIR}/${PROJ}.train.src.char \
                         -t ${TAR_DIR}/${PROJ}.train.tgt.char \
                         -o ${TAR_DIR}/${PROJ}.train.pkl > /dev/null 2>&1

echo -e " # Preprocessing - dev"
python ${MK_SR_DATA__PY} -v ${TAR_DIR}/${PROJ}.vocab_word.txt \
                         -s ${TAR_DIR}/${PROJ}.test.src.char  \
                         -t ${TAR_DIR}/${PROJ}.test.tgt.char  \
                         -o ${TAR_DIR}/${PROJ}.dev.pkl > /dev/null 2>&1

echo -e " # Preprocessing - test"
python ${MK_SR_DATA__PY} -v ${TAR_DIR}/${PROJ}.vocab_word.txt \
			 -s ${TAR_DIR}/${PROJ}.test.src.char  \
		 	 -t ${TAR_DIR}/${PROJ}.test.tgt.char  \
			 -o ${TAR_DIR}/${PROJ}.test.pkl > /dev/null 2>&1

gzip ${TAR_DIR}/*.pkl
