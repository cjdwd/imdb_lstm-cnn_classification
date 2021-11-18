#!/bin/bash
lines=100
if [ "$1" == "word2vec" ]
then
logfile="./logs/word2vec.out"

elif [ "$1" == "tfidf" ]
then
logfile="./logs/tfidf.out"

elif [ "$1" == "tf" ]
then
logfile="./logs/tf.out"
fi

tail -f -n $lines $logfile

