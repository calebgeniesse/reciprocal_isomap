# Reciprocal Isomap

[Reciprocal Isomap](https://calebgeniesse.github.io/reciprocal_isomap) is a reciprocal variant of Isomap for robust non-linear dimensionality reduction in Python. 

The `ReciprocalIsomap` transformer was inspired by scikit-learn's implementation of Isomap, but the reciprocal variant enforces shared connectivity in the underlying *k*-nearest neighbors graph (i.e., two points are only considered neighbors if each is a neighbor of the other).


## Related Projects

- [NeuMapper](https://braindynamicslab.github.io/neumapper/) is a scalable Mapper algorithm for neuroimaging data analysis. The Matlab implementation was designed specifically for working with complex, high-dimensional neuroimaging data and produces a shape graph representation that can be annotated with meta-information and further examined using network science tools.



## Setup

### Dependencies

#### [Python 3.6+](https://www.python.org/)

#### Required Python Packages
* [numpy](https://www.numpy.org)
* [scipy](https://www.scipy.org/)
* [scikit-learn](https://scikit-learn.org)
* [networkx](https://networkx.github.io)


### Install using pip

Assuming you have the required dependencies, you should be able to install using pip.
```bash
pip install git+https://github.com/calebgeniesse/reciprocal_isomap.git
```

Alternatively, you can also clone the repository and build from source. 
```bash
git clone git@github.com:calebgeniesse/reciprocal_isomap.git
cd reciprocal_isomap

pip install -r requirements.txt
pip install -e .
```





## Usage

### The `ReciprocalIsomap` object

Here, we will walk through a simple example using `ReciprocalIsomap`.

```python
import numpy as np 
from reciprocal_isomap import ReciprocalIsomap

# generate some random data points
X = np.random.random((100, 10))

# configure the ReciprocalIsomap object
r_isomap = ReciprocalIsomap(n_neighbors=8)

# fit and transform the data in a single step
embedding = r_isomap.fit_transform(X)
```

We can also fit the data, and then project points into the embedding.

```python
# fit the data and then transform a subset of the data
embedding = r_isomap.fit(X).transform(X[1::2])
```

Similarly, we can project new unseen data points into the embedding.

```python
# fit a subset of the data and then transform the other half of the data 
embedding = r_isomap.fit(X[0::2]).transform(X[1::2])
```




### Comparison with `Isomap`

Here, we compare embeddings created using `Isomap` and `ReciprocalIsomap` across several values of the `n_neighbors` parameter.


First, let's plot the embeddings from `Isomap`.

```python
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

try_n_neighbors = [3, 4, 5, 6]

fig, axes = plt.subplots(1, 4, figsize=(24, 5))
for i,n_neighbors in enumerate(try_n_neighbors):
    isomap = Isomap(n_neighbors=n_neighbors)
    embedding = isomap.fit_transform(X)
    axes[i].scatter(embedding[:,0], embedding[:,1], c=y, cmap='Spectral_r')
    axes[i].set_title(f"Isomap(n_neighbors={n_neighbors})", fontweight='bold') 
```

<p align="center">
<a href="https://github.com/calebgeniesse/reciprocal_isomap/">
<img src="https://github.com/calebgeniesse/reciprocal_isomap/blob/main/examples/isomap_cme_subject_7.png?raw=true">
</a>
</p>



Now, let's plot the embeddings from `ReciprocalIsomap`.

```python
from reciprocal_isomap import ReciprocalIsomap
import matplotlib.pyplot as plt

try_n_neighbors = [3, 4, 5, 6]

fig, axes = plt.subplots(1, 4, figsize=(24, 5))
for i,n_neighbors in enumerate(try_n_neighbors):
    r_isomap = ReciprocalIsomap(n_neighbors=n_neighbors)
    embedding = r_isomap.fit_transform(X)
    axes[i].scatter(embedding[:,0], embedding[:,1], c=y, cmap='Spectral_r')
    axes[i].set_title(f"ReciprocalIsomap(n_neighbors={n_neighbors})", fontweight='bold')
```

<p align="center">
<a href="https://github.com/calebgeniesse/reciprocal_isomap/">
<img src="https://github.com/calebgeniesse/reciprocal_isomap/blob/main/examples/r_isomap_cme_subject_7.png?raw=true">
</a>
</p>





## **Citation**

If you find Reciprocal Isomap useful, please consider citing:
> Geniesse, C., Chowdhury, S., & Saggar, M. (2022). [NeuMapper: A Scalable Computational Framework for Multiscale Exploration of the Brain's Dynamical Organization](https://doi.org/10.1162/netn_a_00229). *Network Neuroscience*, Advance publication. doi:10.1162/netn_a_00229




