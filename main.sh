#!/bin/bash
if [ "$1" == "word2vec" ]
then
logfile="./logs/word2vec.out"
trainfile="main_word2vec.py"
nohup python -u ${trainfile} > ${logfile} 2>&1 &

elif [ "$1" == "tfidf" ]
then
logfile="./logs/tfidf.out"
trainfile="main_tfidf.py"
nohup python -u ${trainfile} $1 > ${logfile} 2>&1 &

elif [ "$1" == "tf" ]
then
logfile="./logs/tf.out"
trainfile="main_tfidf.py"
nohup python -u ${trainfile} $1 > ${logfile} 2>&1 &
fi


