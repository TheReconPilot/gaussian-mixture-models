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

Given a target density $\pi(x_1, \cdots, x_d)$ we sample through sampling from $\pi(x_i | x{-i})$ to update the $i^{th}$ component.

If our current state is $(x_n, y_n, z_n)$ at the $n^{th}$ iteration, then we update our parameter values withthe following steps:

1) Sample $x_{n+1} \sim \pi(x | y_n, z_n)$
2) Sample $y_{n+1} \sim \pi(y | x_{n+1}, z_n)$ 
3) Sample $z_{n+1} \sim \pi(x | x_{n+1}, y_{n+1})$ 
 
At each step of the sampling we use the most recent values of all the other components in the full conditional distribution. {cite}`geman1984` showed that if  $x$  is the density of our parameters at the $n^{th}$ iteration, as $n \rightarrow \infty$ then $p(x_n, y_n, z_n) \rightarrow p(x,y,z)$.

### Example: Gibbs Sampler for unknown $\mu$ and $\sigma$

First we start by recalling that a gaussian mixture model has the following form:

$$
p(x|\theta) = \sum_i \pi_i \phi_{\theta_i}
$$

where, 

$$
\begin{align*}
\phi_{\theta_i}(x) & \sim N(\mu_i, \sigma^2_i) \\
\pi_i & = \text{weight/proportion of $i^{th}$ normal}
\end{align*}
$$

We can now define our prior distributions. We’ll use conjugate priors because they allow us to easily compute posterior distributions. We should also point out that the choice of prior hyper parameters can make our calculations easier as well. We define our priors over $\mu_j,\sigma^2_j,\pi$ as follows:

$$
\begin{align*}
p(\pi) & \sim Dir(\alpha)\\
p(\mu_j) & \sim N(\mu_0 = 0, \tau^2 = 1)\\
p(\sigma_j^2) & \sim IG(\delta = 1, \psi = 1)
\end{align*}
$$

#### Full Conditional for $\mu$:

$$
\begin{align*}
p(\mu|x, z) & = \int_0^{\infty}\int_0^{\infty}p(\theta|x,z)d\pi d\sigma\\
& \propto \prod_{n=1}^N\prod_{j=1}^K\phi_{\theta_j}(x_i)^{z_j}\prod_{j=1}^K\exp\left[-\frac{\mu_j^2}
{2}\right]\\
\end{align*}
$$

We can stick with a singular instance of $\mu$ to simplify this a bit and get rid of the product over $K$ because we know that the calculation is going to be the same for all $\mu$.

$$
\begin{align*}
& \propto \prod_{n=1}^N\phi_{\theta_1}(x_i)^{z_1}\exp\left[-\frac{\mu_1^2}{2}\right]\\
& \propto \exp\left[-\frac{\sum_{i=1}^Nz_{i1}(x_i - \mu_1)^2}{2\sigma_j^2} - \frac{\mu_1^2}{2}\right]\\
& \propto \exp\left[-\frac{\sum_{i=1}^Nz_{i1}x_i^2 - 2\mu_1x_iz_{i1} + z_{i1}\mu_1^2}{2\sigma_j^2} - \frac{\mu_1^2}{2}\right]\\
p(\mu | x, z) & \propto \exp\left[-\frac{\sum_{i=1}^Nz_{i1}x_i^2 - 2\mu_1x_iz_{i1} + z_{i1}\mu_1^2 + \sigma^2_j\mu_j^2}{2\sigma_j^2}\right]
\end{align*}
$$

Now let $\sum_{i=1}^Nz_{ij}x_i=\tilde{x_j}$ and $\sum_{i=1}^Nz_{ij}=n_j$. We can also see that the first 
term $\sum_{i=1}^Nz_{i1}x_i^2$ does not depend on $\mu_j$ so this can be factored out and absorbed into the constant term. We're going to need to complete the square here to isolate our normal parameters.

$$
\begin{align*}
p(\mu | x, z) & \propto \exp\left[-\frac{2\tilde{x_j}\mu_j + (n_j + \sigma^2_j)\mu_j^2}{2\sigma_j^2}\right]\\
& \propto \exp\left[-(n_j + \sigma^2_j)\frac{\mu_j^2 + 2\left(\frac{\tilde{x_j}}{n_j + \sigma^2_j}\right)\mu_j - \left(\frac{\tilde{x_j}}{n_j + \sigma^2_j}\right)^2 + \left(\frac{\tilde{x_j}}{n_j + \sigma^2_j}\right)^2}{2\sigma_j^2}\right]\\
& \propto \exp\left[-(n_j + \sigma^2_j)\frac{\left(\mu_j - \frac{\tilde{x_j}}{n_j + \sigma^2_j}\right)^2}{2\sigma_j^2}\right]\\
p(\mu | x, z) & \sim N\left(\frac{\tilde{x_j}}{n_j + \sigma^2_j}, \frac{\sigma^2_j}{n_j + \sigma^2_j}\right)
\end{align*}
$$

Note that if we use the prior 
$$p(\mu_j|\mu_0,\tau^2) = N(0, \sigma^2_j)$$ we get:

$$
\begin{align*}
p(\mu | x, z) \sim N \left(\frac{\tilde{x_j}}{n_j + 1}, \frac{\sigma^2_j}{n_j + 1}\right)\\
\end{align*}
$$

#### Full Conditional for $\sigma^2$:

Moving on to $\sigma$:

$$
\begin{align*}
p(\sigma^2|x, z) & = \int_0^{\infty}\int_0^{\infty}p(\theta|x,z)d\pi d\mu\\
& \propto \prod_{n=1}^N\prod_{j=1}^K\phi_{\theta_j}(x_i)^{z_j}\prod_{j=1}^K \left(\sigma^2_j\right)^{-2}\exp\left[-\frac{1}{\sigma^2_j}\right]
\end{align*}
$$

Again we can isolate to $j=1$ knowing that it's the same for all $j$:

$$
\begin{align*}
& \propto \prod_{n=1}^N\phi_{\theta_j}(x_i)^{z_j}\left(\sigma^2_j\right)^{-2}\exp\left[-\frac{1}{\sigma^2_j}\right]\\
& \propto \left(\sigma^2_j\right)^{-\frac{\left(\sum_{i=1}^Nz_{i,j}\right) -2 -2}{2}}\exp\left[-\frac{1}{\sigma^2_j}- \frac{\sum_{i=1}^N(x-\mu_j)^2}{2\sigma^2_j}\right]\\
& \propto \left(\sigma^2_j\right)^{-\left(\frac{1}{2}n_j + 1\right) - 1}\exp\left[\frac{1 + \frac{1}{2}\sum_{i=1}^N(x-\mu_j)^2}{\sigma^2_j}\right]\\
& \sim IG\left(\frac{1}{2}n_j + 1, 1 + \frac{1}{2}\sum_{i=1}^N(x-\mu_j)^2\right)
\end{align*}
$$

##### Full conditional for $\pi$:

<!-- $$
\begin{align*}
p(\pmb{\pi}) &amp; \sim Dir(\pmb{\alpha})\\
p(\mu_j) &amp; \sim N(\mu_0 = 0, \tau^2 = 1)\\
p(\sigma_j^2) &amp; \sim IG(\delta = 1, \psi = 1)
\end{align*}
$$ -->


$$
\begin{align*}
% p(\pi|x, z, \pmb{\sigma}, \pmb{\mu}) &amp; \propto \prod_{j=1}^K \pi_j^{\alpha_j - 1 + \sum_{i=1}^N z_i}\\
p(\pi|x, z) & \sim Dir\left(\sum_{i=1}^Nz_1 + \alpha_1, ..., \sum_{i=1}^Nz_k + \alpha_k\right) 
\end{align*}\
$$

## References

```{bibliography}
:filter: docname in docnames
```