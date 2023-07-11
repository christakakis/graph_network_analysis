# Production and evaluation of Node Embeddings
## **Link Prediction and K-Means Clustering.**

#### Working with [**polbooks**](http://networkdata.ics.uci.edu/data/polbooks/) network, a directed graph from Books about US Politics Dataset, we produce node embeddings using Node2Vec and then evaluate their performance using them for Link Prediction and K-Means Clustering, with respect to the ground-truth communities given.

## We'll apply all methods to the largest connected component of the graph and produce node embeddings (vector representations) using:

  • **(i)** Node2Vec with q=2 and p=1
  
  • **(ii)** Node2Vec with q=0.5 and p=1
  
  • **(iii)** Node2Vec with q=1 and p=1
  
### In all experiments we'll give 64 for embedding dimension. All the other parameters (num_wakls, workers, window, min_count, etc...) are kept stable and we can see the final values in the [Python Notebook]() file.


## To evaluate the performance of embeddings we'll use them in the following problems:

  • **(i) Link Prediction**. We measure Accuracy/Precision/Recall for Decision Trees, Naive Bayes, Gradient Boosting, k-NN and SVM.
  
  • **(ii) Clustering**. We'll apply K-Means to the embeddings assuming K=3. We then measure Modularity and Purity. The [t-SNE method](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) is used for visualizing the vector representations of nodes in two dimensions, in order to compare the clustering results with the actual category of nodes - ground truth.


## Finally

We'll summarize our results and comment them in the [Report](https://github.com/christakakis/network_analysis/blob/main/productionAndNetworkMeasurements/Production%20and%20Network%20Measurements%20Report.pdf) file.

This repository was initially created to store my personal python codes but also be available to others trying to build or understand something similar.
The codes contained in this repo are made specifically for a Network Analysis and Web Mining course of my MSc program.

The .txt files that contain the networks used belong to their respective owners linked in this README file. 