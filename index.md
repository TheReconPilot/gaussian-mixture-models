# Gaussian Mixture Models

Within a population, there may be normally distributed subpopulations. A Gaussian Mixture Model (GMM) can be used to represent them, and find these subpopulations.

GMMs can be seen as an extension to K-Means Clustering.

In K-Means, we have some data and we want to identify the clusters in this data population. K-Means Clustering assigns each point to a single cluster, and we get the cluster centers as a result. However, this is a hard clustering method. Each point only belongs to a single cluster.

GMM is a soft clustering method. Instead of assigning a point to a single cluster, we assign probabilities for a point belonging to each of the clusters. GMMs assume the data has normally distributed subpopulations, so each cluster is modelled as a Gaussian. The mixture of these Gaussians is what forms the GMM.

```{tableofcontents}
```

---

This site is the result of a Semester Project on Gaussian Mixture Models by Goirik Chakrabarty and Purva Parmar, under Prof Leelavati Narlikar at IISER Pune.
