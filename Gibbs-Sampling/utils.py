import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def proportion(n, k):
    pi = np.zeros(k)     # Proportion of each class in the mixture

    p = np.random.randint(n, size=k-1) # Partition for data points to get proportions 
    p.sort()
    p = np.append(0, p)
    p = np.append(p, n)

    for i in range(k):
        pi[i] = (p[i+1] - p[i])/n

    return pi

def gen_data(n, k, lamda, sigma_sq, pi=None, mu=None, show_fig=True):

    if pi is None:
        pi = proportion(n, k)                  # Proportion of each class in the mixture
        
    if mu is None:
        mu = np.random.normal(0,lamda,size=k)  # List of means for each class

    sample = np.zeros(n)
    cat_list = np.zeros(n, dtype=int)
    for i in range(n):
        catgory = np.random.choice(np.array(range(k)), p = pi)
        sample[i] = np.random.normal(mu[catgory], sigma_sq)
        cat_list[i] = np.int(catgory)

    df = pd.DataFrame({"cat":cat_list, "sample":sample})

    if show_fig:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

        sns.histplot(data=df, x="sample", hue="cat", kde=True, palette="tab10", ax=ax[0])
        sns.histplot(df["sample"], kde=True, ax=ax[1])
        plt.show()

    return df, pi, mu

def id_to_pi(v):
    n = len(v)
    k = len(v.unique())
    pi = np.zeros(k)
    for i in range(k):
        pi[i] = (v == i).sum()/n

    return pi