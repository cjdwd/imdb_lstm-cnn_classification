### IMDB-text-classification-sentiment-prediction
Easy to start.   
Use deep nerual network to predict the sentiment of movie review.   
Various methods, word2vec, tf-idf and df to generate text vectors.   
Various models including lstm and cov1d.   
Achieve f1 score 92.    
*********************************  
1. Run python text_wash.py in the data_process folder  
2. Run python data_process_word2vec.py, python data_process_tf_manual.py, python data_process_tfidf.py under the data_process folder, this step takes a long time  
3. Use bash main.sh word2vec to perform text classification training using word2vec word vectors as training data, and use bash view.sh word2vec to view the redirected output during training  
4. Use bash eval.sh word2vec to test the saved model after training  
The above word2vec can be replaced by (word2vec/tf/tfidf)  

# Enjoy your training!  


