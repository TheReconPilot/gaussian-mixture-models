# Gibbs Sampling

## Motivation

Calculating a quantity from a probabilistic model is referred to more generally as probabilistic inference, or simply inference.

For example, we may be interested in calculating an expected probability, estimating the density, or other properties of the probability distribution. This is the goal of the probabilistic model, and the name of the inference performed often takes on the name of the probabilistic model, e.g. Bayesian Inference is performed with a Bayesian probabilistic model.

The direct calculation of the desired quantity from a model of interest is intractable for all but the most trivial probabilistic models. Instead, the expected probability or density must be approximated by other means.

> For most probabilistic models of practical interest, exact inference is intractable, and so we have to resort to some form of approximation.

- Pattern Recognition and Machine Learning, 2006 {cite}`bishop2006pattern`

## Background

### Monte Carlo

The solution to the aboce problem is to draw independent samples from the probability distribution, then repeat this process many times to approximate the desired quantity. This is called Monte Carlo sampling.

The problem with Monte Carlo sampling is that it does not work well in high-dimensions:-

1) The curse of dimensionality, where the volume of the sample space increases exponentially with the number of parameters (dimensions).

2) This is because Monte Carlo sampling assumes that each random sample drawn from the target distribution is independent and can be independently drawn. This is typically not the case or intractable for inference with Bayesian structured or graphical probabilistic models.

### Monte Carlo Markov Chain

To deal with the above drawbacks of Monte Carlo methods, Markov Chain Monte Carlo (MCMC) is used for performing inference for probability distributions where independent samples from the distribution cannot be drawn, or cannot be drawn easily.

Samples are drawn from the probability distribution by constructing a Markov Chain, where the next sample that is drawn from the probability distribution is dependent upon the last sample that was drawn. The idea is that the chain will settle on (find steady state) on the desired quantity we are inferring.

## Gibbs Sampling

The Gibbs Sampling algorithm is an approach to constructing a Markov chain where the probability of the next sample is calculated as the conditional probability given the prior sample.

Given a target density $\pi(x_1, \hdots, x_d)$ we sample through sampling from $\pi(x_i | x{-i})$ to update the $i^{th}$ component.

If our current state is $(x_n, y_n, z_n)$ at the $n^{th}$ iteration, then we update our parameter values withthe following steps:

1) Sample $x_{n+1} \sim \pi(x | y_n, z_n)$
2) Sample $y_{n+1} \sim \pi(y | x_{n+1}, z_n)$ 
3) Sample $z_{n+1} \sim \pi(x | x_{n+1}, y_{n+1})$ 
 
At each step of the sampling we use the most recent values of all the other components in the full conditional distribution. {cite}`geman1984` showed that if  $x$  is the density of our parameters at the $n^{th}$ iteration, as $n \rightarrow \infty$ then $p(x_n, y_n, z_n) \rightarrow p(x,y,z)$.

### Example: Gibbs Sampler for unknown $\mu$ and $\sigma$



## References

```{bibliography}
:filter: docname in docnames
```