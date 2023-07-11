# Graph Network Analysis
## **Study, analysis and extraction of knowledge from the web and social networks.**

#### In this repo we study, analyse and extract knowledge from the web and from different social network graphs.

## Briefly, this repo contains:

  ## [productionAndNetworkMeasurements](https://github.com/christakakis/graph_network_analysis/tree/main/productionAndNetworkMeasurements)
  • **Study and analyse social networks.** Compare the properties of the networks, comment on our results and our parameter choices for the synthetic networks, and also the reasoning we used to arrive at them. Also we do some plotting.

  ## [node2VecClusteringLinkPrediction](https://github.com/christakakis/graph_network_analysis/tree/main/node2VecClusteringLinkPrediction)
  • **Production and evaluation of Node Embeddings.** Working with [**polbooks**](http://networkdata.ics.uci.edu/data/polbooks/) network, a directed graph from Books about US Politics Dataset, we produce node embeddings using Node2Vec and then evaluate their performance using them for **Link Prediction** and K-Means **Clustering**, with respect to the ground-truth communities given.
  
  ## [communityDetectionTechniques](https://github.com/christakakis/graph_network_analysis/tree/main/communityDetectionTechniques)
  • **Detect communities with different techniques.** Working with [**polblogs**](http://wwwpersonal.umich.edu/~mejn/netdata/polblogs.zip) network, a directed graph from hyperlinks between blogs on US politics recorded in 2005 by Adamic and Glance, we'll try to apply and compare the performance of different community detection techniques, with respect to the ground-truth communities given.

**The files that contain the networks used in this repository belong to their respective owners.**

This repository was initially created to store my personal python codes but also be available to others trying to build or understand something similar.
The codes contained in this repo are made specifically for a Network Analysis and Web Mining course of my MSc program.
