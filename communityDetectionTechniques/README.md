# Production And Network Measurements
## **Problem 1 | Detect communities with different techniques.**

#### Working with [**polblogs**](http://wwwpersonal.umich.edu/~mejn/netdata/polblogs.zip) network, a directed graph from hyperlinks between blogs on US politics recorded in 2005 by Adamic and Glance, we'll try to apply and compare the performance of different community detection techniques, with respect to the ground-truth communities given.

## We'll apply and compare the performance of the following community detection techniques:

  • **(1) Cliques Finding.** A method based on finding a clique.
  
  • **(2) Modularity Maximazation.** A method of maximizing modularity.
  
  • **(3) Agglomerative.** An agglomerative hierarchical method.
    
  • **(4) Hierarhical Clustering - Girvan Newman.** A divisive hierarchical method.
      
  • **(5) Spectral Clustering.** A spectral analysis method.
  

## For each of the methods we measure the following:

  • **(i)** Precision.
  
  • **(ii)** Recall.
  
  • **(iii)** Purity.
  
  • **(iv)** NMI.
  
  • **(v)** Modularity (and compare with the modularity of ground-truth communities).
  
  • **(vi)** Conductance (and compare with the conductance of ground-truth communities).
  
  • **(vii)** Density (and compare with the density of ground-truth communities).
 
 
## Furthermore we explain why we used each of the methods and we do some interesting plots of the resulting communities as well as the ground-truth communities.

## **Problem 2 | Production of Synthetic Networks.**
