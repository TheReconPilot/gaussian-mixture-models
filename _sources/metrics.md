# Metrics

There exist various metrics to compare performance of a clustering.

We are performing experiments where we are generating the data, so we also know the ground truth, i.e. what cluster each point originally belongs to. In such cases, Rand Index (RI) and Adjusted Rand Index (ARI) provide for a good comparison metric.

## Rand Index

$$
    \textsf{Rand Index} = \frac{\textsf{Number of agreeing pairs}}{\textsf{Number of pairs}}    
$$

We simply compare the original clustering with the predicted clustering, and see how many of those agree.

## Adjusted Rand Index

The raw RI score can be adjusted _for chance_.

$$
\textsf{Adjusted Rand Index} = \frac{\textsf{RI - Expected RI}}{\textsf{Max RI - Expected RI}}
$$

Consider all the possible pairings (original and possible predictions). The maximum RI of all of these pairings form the $\textsf{Max RI}$. The expected RI of all these pairings forms the $\textsf{Expected RI}$.