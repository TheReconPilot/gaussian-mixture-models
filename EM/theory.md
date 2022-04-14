# Theory - EM

The source for most of the theory here is *Machine Learning: A Probablistic Perspective* by Kevin P. Murphy {cite}`murphy2012`.

Let us represent all the parameters generally as $\boldsymbol \theta$.

$$
\boldsymbol \theta = \{\boldsymbol \pi, \boldsymbol \mu, \boldsymbol \Sigma\}
$$

## The Need for EM

For computing Maximum Likelihood Estimate (MLE) or Maximum A Posteriori (MAP) Estimate, one approach is to use a generic gradient-based optimizer to find a local minimum of the Negative Log Likelihood:

$$
\text{NLL}(\boldsymbol \theta) = - \frac{1}{N} \log P(\boldsymbol X | \boldsymbol \theta)
$$

However, there are constraints that need to be enforced, like
- Covariance matrices must be positive semi-definite
- Proportions ($\pi_k$) must sum to one

This can be tricky. It can be simpler to use an iterative algorithm like Expectation Maximization (EM).

EM alternates between inferring the missing values given the parameters (the E step), and then optimizing the parameters given the "filled in" data (the M step).


## EM in GMM

### Initialization

We first initialize our parameters. We can use the results of K-Means Clustering as a starting point.

### The E Step

Consider the Auxiliary Function of expected complete data log likelihood:

$$
Q(\boldsymbol \theta^*, \boldsymbol \theta) = \mathbb{E}[\log P(\boldsymbol X, \boldsymbol Z | \boldsymbol \theta^*)] = \sum_{\boldsymbol Z} P(\boldsymbol Z | \boldsymbol X, \boldsymbol \theta) \log P(\boldsymbol X, \boldsymbol Z | \boldsymbol \theta^*)
$$

Out of this, $P(\boldsymbol Z | \boldsymbol X, \boldsymbol \theta)$ is just $\gamma(z_{nk})$, as seen in Equation {eq}`responsibilities`.

And for the other part, use Equations {eq}`P(z)` and {eq}`P(x|z)`.

$$
\begin{aligned}
P(\boldsymbol X, \boldsymbol Z | \boldsymbol \theta^*) &= P(\boldsymbol X | \boldsymbol Z, \boldsymbol \theta) P(\boldsymbol Z | \boldsymbol \theta) = \prod_{n=1}^{N} \prod_{k=1}^{K} \mathcal{N} (\boldsymbol x_n | \boldsymbol \mu_k, \boldsymbol \Sigma_k)^{z_{nk}} \pi_k^{z_{nk}} \\ \\
\implies \log P(\boldsymbol X, \boldsymbol Z | \boldsymbol \theta^*) &= \sum_{n=1}^N \sum_{k=1}^K z_{nk} \left[ \log \pi_k + \log \mathcal{N}(\boldsymbol x_n | \boldsymbol \mu_k, \boldsymbol \Sigma_k) \right]
\end{aligned}
$$

Since the latent variable is only 1 once anytime we evaluate the summation, our final auxiliary function becomes:

```{math}
:label: the-auxiliary-function
Q(\boldsymbol \theta^*, \boldsymbol \theta) = \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) \left[ \log \pi_k + \log \mathcal{N}(\boldsymbol x_n | \boldsymbol \mu_k, \boldsymbol \Sigma_k) \right]
```

The E Step is about computing the missing values.

In practice, however, we only need to compute the responsibilities $\gamma(z_{nk})$ in the E step. The auxiliary function in its full form is required to get some results in the M step.

### The M Step

The M step is about using the "filled in data" in the E step and the existing parameters $\boldsymbol \theta$ to compute revised parameters $\boldsymbol \theta^*$ such that:

$$
\boldsymbol \theta^* = \underset{\boldsymbol \theta}{\text{argmax}}\ Q(\boldsymbol \theta^*, \boldsymbol \theta)
$$

For $\boldsymbol \pi$, we have:

```{math}
:label: optimal-pi
\pi_k^* = \frac{1}{N} \sum_{n=1}^{N} \gamma(z_{nk})
```

For the revised values of mean and covariances, we compute the partial derivatives of $Q$ with respect to these parameters and equate to zero. One can show that the new parameter estimates are given by:

```{math}
:label: optimal-mu
\boldsymbol \mu_k^* = \frac{\sum_{n=1}^N \gamma(z_{nk}) \boldsymbol x_n}{\sum_{n=1}^N \gamma(z_{nk})}
```

```{math}
:label: optimal-sigma
\boldsymbol \Sigma_k^* = \frac{\sum_{n=1}^N \gamma(z_{nk}) (\boldsymbol x_n - \boldsymbol \mu_k) (\boldsymbol x_n - \boldsymbol \mu_k)^\intercal}{\sum_{n=1}^N \gamma(z_{nk})}
```

Computing these revised parameters constitutes the M step.

---

## References

```{bibliography}
:filter: docname in docnames
```