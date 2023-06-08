# Production And Network Measurements
## **Study and analyse social networks.**

#### Compare the properties of the networks, comment on our results and our parameter choices for the synthetic networks, and also the reasoning we used to arrive at them.

#### We'll study the following networks:

  • **(1) [Gnutella](https://snap.stanford.edu/data/p2p-Gnutella05.html).** Peer-to-peer network.
  
  • **(2) [GrQc](https://snap.stanford.edu/data/ca-GrQc.html).** Scientific Collaboration Network.
  
  • **(3) Erdős-Renyi.** Random graph model - Undirected graph.
    
  • **(4) Watts and Strogatz.** Uundirected graph.
      
  • **(5) Barabasi-Albert model.** Undirected graph.
        
  • **(6) Barabasi-Albert.** Non-directional graph.

For the generation of synthetic networks, we chose parameters so that the networks have corresponding properties to the network (2).

#### For each network we measure the following:

  • **(a)** the number of distinct nodes
  
  • **(b)** the number of nodes with self-loop
  
  • **(c)** the number of edges in the network
  
  • **(d)** the number of reciprocated edges
  
  • **(e)** the number of sink and source nodes
  
  • **(f)** maximum, minimum and average degree
  
  • **(g)** maximum, minimum and average degree of incoming edges
  
  • **(h)** maximum, minimum and average degree of outgoing edges
  
  • **(i)** the diameter of the network
  
  • **(j)** the average clustering coefficient and the global clustering coefficient
  
  • **(k)** the size of the largest strongly coherent component in terms of number of nodes, and edges
  
  • **(l)** the size of the largest weakly coherent component in number, nodes, and edges

#### Furthermore plot the degree distribution of the nodes using:

  • **(m)** simple distribution and linear scale
  
  • **(n)** simple distribution and log-log scale
  
  • **(o)** distribution using exponentially increasing size bins (log bininning) on a logarithmic scale
 
Compare the properties of the networks, comment on your results and the choices of
your parameter choices for the synthetic networks, and on what rationale you arrived at them.


Then we'll summarize our results and comment them, compare the properties of the networks, also comment the choices of our parameters for the synthetic networks, and on what justification we arrived at them. in the [Report](https://github.com/christakakis/network_analysis/blob/main/productionAndNetworkMeasurements/Production%20and%20Network%20Measurements%20Report.pdf) file.

This repository was initially created to store my personal python codes but also be available to others trying to build or understand something similar.
The codes contained in this repo are made specifically for a Network Analysis and Web Mining course of my MSc program.
The .txt files that contain the networks used belong to their respective owners linked in this README file. 
