#!/bin/bash
if [ "$1" == "word2vec" ]
then
evalfile="eval_word2vec.py"
python -u ${evalfile}

elif [ "$1" == "tfidf" ]
then
evalfile="eval_tfidf.py"
python -u ${evalfile} $1

elif [ "$1" == "tf" ]
then
evalfile="eval_tfidf.py"
python -u ${evalfile} $1
fi


