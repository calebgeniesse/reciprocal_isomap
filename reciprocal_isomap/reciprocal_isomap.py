"""
Reciprocal Isomap for robust non-linear dimensionality reduction.

Authors: Caleb Geniesse, geniesse@stanford.edu
         Samir Chowdhury, samirc@stanford.ed
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.validation import check_is_fitted, check_symmetric
from sklearn.utils.graph import graph_shortest_path
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer

from scipy import sparse
from scipy.sparse import block_diag, csr_matrix, issparse, csgraph

import networkx as nx
from networkx.algorithms.shortest_paths.dense import floyd_warshall_numpy as fw

__all__ = [
    'ReciprocalIsomap'
]


class ReciprocalIsomap(BaseEstimator, TransformerMixin):
    """ Reciprocal Isomap Embedding 

        A reciprocal variant of Isomap for robust non-linear dimensionality 
        reduction in Python.

        Parameters
        ----------
        n_neighbors : int, number of neighbors to use

        n_components : int, number of components to use for embedding

        distance_mode : str, method for computing geodesic distances

            Possible inputs:
            - 'csgraph' uses ``scipy.sparse.csgraph.shortest_path``
            - 'landmark' uses ``scipy.sparse.csgraph.dijkstra`` to landmarks
            - 'geodesic' uses ``sklearn.utils.graph.graph_shortest_path`` 
            - 'geodesic_deprecated' computes floyd_warshall_numpy for each subgraph

        neighbors_mode : str, type of weights to use for nearest neighbor graph

            Possible inputs:
            - 'connectivity' uses binary neighbor connectivity as edge weights
            - 'distance' uses neighbor distances as edge weights
        
        # TODO
        #  neighbors_estimator : 
        #  neighbors_algorithm : 
        #  metric : 
        #  p : 
        #  metric_params : 

        n_jobs : int or None, optional (default=None)
            Number of CPUs to use for NearestNeighbors and KernalPCA
            
            Possible inputs:
            - ``None`` means 1 unless in a `joblib.parallel_backend` context.
            - ``-1`` means using all processors.
            - ``-2`` means using all but 1 processors.


        Examples
        --------
        >>> from reciprocal_isomap import ReciprocalIsomap
        >>> import numpy as np 
        >>> X = np.random.random((100, 10))
        >>> r_isomap = ReciprocalIsomap(n_neighbors=8)
        >>> embedding = r_isomap.fit_transform(X)

    """

    def __init__(self, 
                 n_neighbors=5, 
                 n_components=2, 
                 distance_mode="geodesic",  # geodesic, csgraph, landmark, 
                 neighbors_mode="distance", # connectivity or distance
                 neighbors_estimator=None,
                 neighbors_algorithm='auto',
                 metric='minkowski', p=1, 
                 metric_params=None,
                 n_jobs=None,
        ):         
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.distance_mode = distance_mode 
        self.neighbors_mode = neighbors_mode
        self.neighbors_estimator = neighbors_estimator 
        self.neighbors_algorithm = neighbors_algorithm
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs



    def _reciprocal_distances(self, X, landmarks=None, neighbors_estimator=None):
        """ Construct geodesic distance matrix from XA.
        """

        # check neighbors estimator
        if neighbors_estimator is None:
            neighbors_estimator = self.neighbors_estimator
        
        # setup neighbors estimator (if needed)
        if neighbors_estimator is None:
            neighbors_estimator = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                algorithm=self.neighbors_algorithm,
                metric=self.metric, p=self.p,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs
            )

        # fit to the data, if needed
        if not hasattr(neighbors_estimator, 'n_samples_fit_'):
            neighbors_estimator.fit(X)
        elif neighbors_estimator.n_samples_fit_ != X.shape[0]:
            neighbors_estimator.fit(X)
            

        # compute k-nearest neighbor distances (not including self)
        neighbor_distance_matrix = neighbors_estimator.kneighbors_graph(
             X=X, n_neighbors=self.n_neighbors+1, mode="distance"
        )
        neighbor_distance_matrix.data[np.isinf(neighbor_distance_matrix.data)] = 0                                      
        neighbor_distance_matrix.data[np.isnan(neighbor_distance_matrix.data)] = 0 
        neighbor_distance_matrix.eliminate_zeros()
        neighbor_matrix = neighbor_distance_matrix.copy()
        neighbor_matrix.data[:] = 1 # set all non-empty values to 1
     
        # TODO: try using RBF-based kernel or distance cutoff to determine
        #       reciprocal neighbors
        # neighbor_matrix = neighbor_distance_matrix < self.eps


        # check for symmetric distance matrix
        if neighbor_matrix.shape[0] == neighbor_matrix.shape[1]:

            # drop non-reciprocal connections
            neighbor_matrix = neighbor_matrix.multiply(neighbor_matrix.T)
            # check_symmetric(neighbor_matrix)

            #  drop non-reciprocal distances
            neighbor_distance_matrix = neighbor_distance_matrix.multiply(neighbor_matrix)
            # check_symmetric(neighbor_distance_matrix)

        else:
            
            # TODO: figure out how to handle non-reciprocal data
            #       e.g., out-of-sample landmarks
            pass 
        

        # set the edge weights to 'connectivity' (default='distance')
        if self.neighbors_mode == "connectivity":
            neighbor_distance_matrix = neighbor_matrix.copy()


        # construct geodesic distance matrix (slower)
        if self.distance_mode == "geodesic_deprecated":
            
            # compute block diagonal matrix of geodesic distances
            if issparse(neighbor_distance_matrix):
                kng = nx.from_scipy_sparse_matrix(neighbor_distance_matrix)
            else:
                kng = nx.from_numpy_matrix(neighbor_distance_matrix)
            
            # compute fw distances for each connected component
            sd = [fw(kng.subgraph(c)) for c in nx.connected_components(kng)]
           
            # set distance matrix
            geodesic_distance_matrix = block_diag(sd, format='csr')
        
        elif self.distance_mode == "geodesic":
        
            # compute shortest path distances using weight matrix
            geodesic_distance_matrix = graph_shortest_path(
                  neighbor_distance_matrix, method="auto", directed=False
            )

        elif self.distance_mode == "csgraph":

            # compute shortest path distances using weight matrix
            geodesic_distance_matrix = csgraph.shortest_path(
                neighbor_distance_matrix, method='auto', 
                directed=False, unweighted=False, 
            )

        elif self.distance_mode == "landmark":
            
            # check for landmarks 
            if landmarks is None:
                landmarks = np.arange(X.shape[0])

            # only compute distances to the landmarks (i.e., indices=landmarks)
            landmark_distance_matrix = csgraph.dijkstra(
                neighbor_distance_matrix, 
                directed=False, unweighted=False,  
                indices=landmarks,
            )                   
            geodesic_distance_matrix = np.zeros(neighbor_distance_matrix.shape)
            geodesic_distance_matrix[:,landmarks] = landmark_distance_matrix.T 

        else:

            # just use the weighted graph as the distance matrix
            # geodesic_distance_matrix = neighbor_distance_matrix.copy()
            raise Exception(
                f"Distance mode '{self.distance_mode}' is not valid. "
                f"Possible values include {'geodesic', 'csgraph', 'landmark'}."
            )


        # convert to sparse array (if not already)
        if not issparse(geodesic_distance_matrix):
            geodesic_distance_matrix = csr_matrix(geodesic_distance_matrix)
        
        # TODO: figure out how to connect un-connected components?
        # set infs to 0 for now, eliminate zeros
        geodesic_distance_matrix.data[np.isinf(geodesic_distance_matrix.data)] = 0                                      
        geodesic_distance_matrix.data[np.isnan(geodesic_distance_matrix.data)] = 0                                      
        geodesic_distance_matrix.eliminate_zeros()

       
        # store sparse arrays? (skip to reduce memory footprint)
        # self.neighbor_matrix_ = neighbor_matrix
        # self.neighbor_distance_matrix_ = neighbor_distance_matrix
        # self.geodesic_distance_matrix_ = geodesic_distance_matrix
        self.neighbors_estimator_ = neighbors_estimator

        # return geodesic distance matrix
        return geodesic_distance_matrix


    
    def fit(self, X, y=None, landmarks=None):
        """Fit X.
        
        The code below was modified from sklearn.manifold.Isomap._fit_transform
        """
        self.n_samples_fit_ = X.shape[0]
        self.n_features_in_ = X.shape[-1]

        # compute reciprocal nearest neighbors and distances
        self.dist_matrix_ = self._reciprocal_distances(X, landmarks=landmarks)

        # initialize models
        self.kernel_pca_ = KernelPCA(
            n_components=self.n_components,
            kernel="precomputed",
            n_jobs=self.n_jobs
        )       
    
        # convert to isomap kernel
        self.kernel_matrix_ = -0.5 * self.dist_matrix_.A ** 2
        
        # compute embedding vectors for data  
        self.embedding_ = self.kernel_pca_.fit_transform(self.kernel_matrix_)
        
        # return instance
        return self
      
        
    def fit_transform(self, X, y=None, landmarks=None):
        """ Fit X and return the embedding_"""
        return self.fit(X, y=y, landmarks=landmarks).embedding_
  
   
    def transform(self, X, n_neighbors=1, return_distance=False):
        """Transform new X.

        This is implemented by linking the points X into the graph of geodesic
        distances of the training data. First the `n_neighbors` nearest
        neighbors of X are found in the training data, and from these the
        shortest geodesic distances from each point in X to each point in
        the training data are computed in order to construct the kernel.
        The embedding of X is the projection of this kernel onto the
        embedding vectors of the training set.

        Parameters
        ----------
        X : array-like, shape (n_queries, n_features)
            If neighbors_algorithm='precomputed', X is assumed to be a
            distance matrix or a sparse graph of shape
            (n_queries, n_samples_fit).

        Returns
        -------
        X_new : array-like, shape (n_queries, n_components)

        Notes
        -----
        - do we need to change this for reciprocal?
          e.g. using block_diag distances?
        
        """
        check_is_fitted(self)
        
        # let's be conservative with n_neighbors, since we won't be
        # dropping non-reciprocal connections like we did during training
        distances, indices = self.neighbors_estimator_.kneighbors(
            X=X, n_neighbors=n_neighbors, return_distance=True
        )
        
        # TODO test shared radius version
        # filter distances based on radii computed for 
        # k-th reciprocal neighbor of each data point
        # 
        # if new point is not within any of these radii,
        # then label as noise / new landmark? (update embedding?)

        # Create the graph of shortest distances from X to
        # training data via the nearest neighbors of X.
        # This can be done as a single array operation, but it potentially
        # takes a lot of memory.  To avoid that, use a loop:

        n_samples_fit = self.neighbors_estimator_.n_samples_fit_
        n_queries = distances.shape[0]
        G_X = np.zeros((n_queries, n_samples_fit))
        for i in range(n_queries):
            
            if self.neighbors_mode == 'connectivity':
                mask_positive_finite = (distances[i] > 0) & (distances[i] < np.inf)
                distances[i][mask_positive_finite] = 1
                # distances[i][:] = 1

            G_X[i] = np.min(self.dist_matrix_[indices[i]] +
                            distances[i][:, None], 0)
        
        # compute new embedding
        G_X = -0.5 * G_X ** 2
        embedded = self.kernel_pca_.transform(G_X)

        # return embedded, distances (optional)
        if return_distance:
            return embedded, min_distances
        return embedded

