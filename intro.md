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

## References

```{bibliography}
```