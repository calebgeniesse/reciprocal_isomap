# reciprocal_isomap
A reciprocal variant of Isomap for robust non-linear dimensionality reduction in Python. 

Note, the `ReciprocalIsomap` class was inspired by scikit-learn's implementation of `Isomap`. The main difference introduced by `ReciprocalIsomap` is a requirement that neighbors in the underlying *k* nearest neighbors graph must be reciprocal (i.e., two points are only considered neighbors if each is a neighbor of the other).





## Setup

### Dependencies

#### [Python 3.6+](https://www.python.org/)

#### Required Python Packages
* [numpy](https://www.numpy.org)
* [scipy](https://www.scipy.org/)
* [scikit-learn](https://scikit-learn.org)
* [networkx](https://networkx.github.io)


_For a full list of packages, see [`requirements.txt`](https://github.com/calebgeniesse/reciprocal_isomap/blob/master/requirements.txt)._


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

See below for a few examples using `reciprocal_isomap`. 


### The `ReciprocalIsomap` object


Here, we will walk through a simple example using `ReciprocalIsomap`.

```python
from reciprocal_isomap import ReciprocalIsomap
```

First, let's generate some random data points.
```python
import numpy as np 

# create some data
X = np.random.random((100, 10))
```

Next, configure the `ReciprocalIsomap` object.
```python
# initize the model
r_isomap = ReciprocalIsomap(n_neighbors=8)
```

Then simply fit the model and transform your data.
```python
embedding = r_isomap.fit_transform(X)
```

Alternatively, fit your model to the data, and then transform the same data, or project new data into the embedding. 

```python
r_isomap.fit(X)
embedding = r_isomap.transform(X[:10])
```
