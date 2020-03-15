Required packages to run:

gensim==3.8.1
scikit-learn==0.22.1

***The results for task 1 and task 2 are placed in trial_data_public\results.
The training stage may take around 2-3 minutes to complete.
Also due to the randomness of Doc2Vec algorithm, running the program again may produce different embeddings and therefore different distances (which could result in different results for task 1 and task 2)
To combat this I ran the program 10 times, and took the mean of the distances for each word, and used those values for my results.

I chose a threshold distance of 0.4 for task 2.