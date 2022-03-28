# Introduction

Gaussian Mixture Models can be used to represent subpopulations which are normally distributed within an overall population. It is, in essence, a superposition of multiple Gaussians.

We will be considering Multivariate Gaussians, as the univariate case easily follows. The source of most of the material is *Pattern Recognition and Machine Learning* by C. Bishop {cite}`bishop2006pattern`

## Multivariate Normal Distribution

The Multivariate Normal $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ is 

```{math}
:label: multivariate-normal
P(\boldsymbol x | \boldsymbol \mu, \boldsymbol \Sigma) = \frac{1}{\sqrt{\det(2\pi\boldsymbol\Sigma)}} \exp\left(-\frac{1}{2}(\boldsymbol x - \boldsymbol \mu)^\intercal\ \boldsymbol \Sigma^{-1}\ (\boldsymbol x - \boldsymbol \mu)\right)
```

Suppose the number of dimensions is $n$.

Here, $\boldsymbol \mu \in \mathbb{R}^n$ is the **mean**, and $\boldsymbol \Sigma \in \mathbb{R}^{n \times n}$ is the **covariance**, which must be a positive semi-definite matrix.

## Definitions

Assume

- There is a mixture of $K$ Gaussians
- Each Gaussian is in a $D$-dimensional space
- Denote a single $D$-dimensional observation/data point as $\boldsymbol x$
- There are $N$ such observations/points.

Since there are $K$ Gaussians (or **Clusters** from here onwards), we also have the corresponding $K$ means and covariance matrices.

- $\boldsymbol\mu = \{\boldsymbol \mu_1, \boldsymbol \mu_2, \dots, \boldsymbol \mu_K\}$
- $\boldsymbol\Sigma = \{\boldsymbol \Sigma_1, \boldsymbol \Sigma_2, \dots, \boldsymbol \Sigma_K\}$

And the entire data/observation set: $X = \{\boldsymbol x_1, \boldsymbol x_2, \dots, \boldsymbol x_N\}$

A GMM is a mixture of Gaussians. The **proportions** in which these Gaussians are present are represented as $\pi_k$. So, $\pi_k$ is the proportion, also called *mixing probability*, which shows how big the $k^{th}$ Gaussian will be in the mixture.

- $\boldsymbol \pi = \{\pi_1, \pi_2, \dots, \pi_K\}$

The proportions or mixing probabilities satisfy the following:

$$
0 \leqslant \pi_k \leqslant 1\quad \forall\ k \\\\
\sum_{k=1}^{K} \pi_k = 1
$$

## Basic Ideas

We have $N$ observations in a $D$-dimensional space. We want to fit the $K$ Gaussians in a mixture model. So, at the end, we wish to find the parameters of $K$ normally distributed subpopulations in our data/observations and we want to know the probabilities of a point belonging to each of the clusters/gaussians.

### Mixture Distribution

We have K Gaussians, each having a proportion. The mixture distribution can be written as:

```{math}
:label: mixture-dist
P(\boldsymbol x) = \sum_{k=1}^{K} \pi_k\ \mathcal{N}(\boldsymbol x | \boldsymbol \mu_k, \boldsymbol \Sigma_k)
```

where $\mathcal{N}(\boldsymbol x | \boldsymbol \mu_k, \boldsymbol \Sigma_k)$ is the Multivariate Normal described in Equation {eq}`multivariate-normal`.

### Introducing a Latent Variable

Latent Variables are also called Hidden Variables. Here, we introduce $\boldsymbol z$, a $K$-dimensional binary random variable which follows a 1-of-K representation.

For a given observation point, when it belongs to cluster $k$, then $z_k = 1$ and all other $z_i = 0$. So, $z_k \in \{0, 1\}$ and $\sum_k z_k = 1$.

So, $P(z_{k} = 1 | \boldsymbol x)$ would represent the probability of $\boldsymbol x$ belonging to cluster $k$. This is one of the things we wish to find.

The marginal distribution over $\boldsymbol z$ is specified in terms of the mixing probabilities (or proportions) $\pi_k$ such that

$$
p(z_k = 1) = \pi_k
$$

Intuitively, the marginal distribution over $\boldsymbol z$ should represent the probability of cluster $k$, which is exactly what the proportions are.

Because $\boldsymbol z$ uses a 1-of-K representation, we can write this distribution in the form:

$$
P(\boldsymbol z) = \prod_{k=1}^{K} \pi_k^{z_k}
$$

Similarly, the conditional distribution of $\boldsymbol x$ given a particular value for $\boldsymbol z$ is a Gaussian:

$$
P(\boldsymbol x | z_k = 1) = \mathcal{N}(\boldsymbol x | \boldsymbol \mu_k, \boldsymbol \Sigma_k)
$$

which can be written in the form

$$
P(\boldsymbol x | \boldsymbol z) = \prod_{k = 1}^{K} \mathcal{N}(\boldsymbol x | \boldsymbol \mu_k, \boldsymbol \Sigma_k)^{z_k}
$$

This works because only one $z_k = 1$ at a time, and the rest are 0.

The joint probability distribution is $P(\boldsymbol x, \boldsymbol z) = P(\boldsymbol z) P(\boldsymbol x | \boldsymbol z)$, so we can find the marginal distribution of $\boldsymbol x$ by summing over $\boldsymbol z$.

$$
P(\boldsymbol x) = \sum_{z} P(\boldsymbol z)P(\boldsymbol x | \boldsymbol z) = \sum_{k = 1}^{K} \pi_k\ \mathcal{N} (\boldsymbol x | \boldsymbol \mu_k, \boldsymbol \Sigma_k)
$$

For each data point / observation $\boldsymbol x_n$, there is a corresponding latent variable $\boldsymbol z_n$. We have obtained the same formulation of a Gaussian Mixture Model as Equation {eq}`mixture-dist`, this time involving a latent variable.

### Responsibilites

The quantity $P(z_k = 1 | \boldsymbol x)$, which was the probability that the observation $\boldsymbol x$ belongs to cluster $k$, is also denoted as $\gamma(z_k)$.

```{math}
:label: responsibilities
\begin{aligned}
\gamma(z_k) \equiv P(z_k = 1 | \boldsymbol x) &= \frac{P(z_k = 1) P(\boldsymbol x | z_k = 1)}{\sum_{j=1}^{K} P(z_j = 1) P(\boldsymbol x | z_j = 1)} \\ \\
\implies \gamma(z_k) &= \frac{\pi_k\ \mathcal{N}(\boldsymbol x | \boldsymbol \mu_k, \boldsymbol \Sigma_k)}{\sum_{j=1}^{K} \pi_j\ \mathcal{N}(\boldsymbol x | \boldsymbol \mu_j, \boldsymbol \Sigma_j)}
\end{aligned}
```

This quantity is also called the **responsibility** that cluster/gaussian/component $k$ takes for *explaining* the observation $\boldsymbol x$.

## References

```{bibliography}
```