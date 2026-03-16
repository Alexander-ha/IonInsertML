import numpy as np
import pandas as pd

from ase.geometry import find_mic

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm


class BayesianOptimization:
    """
    Bayesian optimization with Gaussian Process Regression for intercalation modeling

    Parameters
    ----------
    int_atom: str, default= 'Li'
        atomic type of the inserted atom(s)

    n_candidates: int, default= 500
        the number of candidates to sample and to be later filtered according to min distances (rmin & rmin_insrt)

    rmin: float, default= 1.45
        minimal distance between host atoms and inserted atoms
        is defined as a sum of atomic radius of the host atom type and ionic raduis of inserted atom type

    rmin_insrt: float, default= 1.34
        minimal distance between inserted atoms
        is defined as a doubled ionic radius of the inserted atom type

    host: np.array, default= None
        an array of host atoms' cartesian coordinates

    rprimd: np.array, default= None
        an array of cell vectors of the host structure

    random_state: int, default= None
        a fixed random state for results reproducibility
    """

    def __init__(self, int_atom='Li', n_candidates=100, rmin=1.45, rmin_insrt=1.34, host=None, rprimd=None, random_state=None, kernel=None, gpr_=None):
        self.int_atom = int_atom
        self.n_candidates = n_candidates
        self.rmin = rmin
        self.rmin_insrt = rmin_insrt
        self.host = host
        self.rprimd = rprimd
        self.kernel = kernel
        self.gpr_ = gpr_
        if self.kernel is None:
            self.kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
        if self.gpr_ is None:
            self.gpr_ = GaussianProcessRegressor(kernel=self.kernel, normalize_y=False)


        if random_state is None:
            self.rng = np.random 
        elif isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = random_state


    def fit(self, X, y):

        """Builds GaussianProcessRegressor

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """

        self.X_ = X.copy()
        self.y_ = y.copy()

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        self.gpr_.fit(X_scaled, y_scaled)

        return self


    def _predict(self, X):

        """Predict mean function (mu) and standard deviation (sigma)
           by Gaussian Process regression.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """

        X_scaled = self.scaler_X.transform(X)

        mu_scaled, sigma_scaled = self.gpr_.predict(X_scaled, return_std=True)

        mu = self.scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()
        sigma = sigma_scaled * self.scaler_y.scale_

        return mu, sigma


    def _acquisition(self, X, xi=0.01, y=None):

        """Acquisition function (expected improvement).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        xi: float, a compromise parameter for exploration/exploitations

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        if y is None:
            y = self.y_

        mu, sigma = self._predict(X)    
        mu_sample_opt = np.min(y)

        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei


    def _generate_candidates(self, n_candidates=None, k=None):
        """
        Generates candidates (X / configurations) for k inserted atoms.

        Parameters
        ----------
        n_candidates : int, optional
            A number of candidates (configurations).
        k : int, optional
            A number of atoms in configuration.
            If None, is defined by self.X_ shape.

        Returns
        -------
        X_candidates : ndarray of shape (n_candidates, 3*k)
            Coordinate vectors of inserted atoms' coordinates.
        """

        if n_candidates is None:
            n_candidates = self.n_candidates
        if k is None:
            k = self.X_.shape[1] // 3


        all_coords = self.rng.uniform(0.0, 1.0, size=(n_candidates * k, 3)) @ self.rprimd
        all_coords = all_coords.reshape(n_candidates, k, 3)
        X_candidates = all_coords.reshape(n_candidates, -1)

        return X_candidates


    def min_image_distance(a, b, cell):
        """
        minimal distance between the two atoms with respect to PBC.

        Parameters
        ----------
        a, b : ndarray or an object .position argument
            Points coordinates. If an object passed (ASE), .positions will be used
        cell : ndarray of shape (3, 3)
            Cell vectors (rprimd).

        Returns
        -------
        float
            minimal distance between given points in a periodic cell
        """

        if hasattr(a, 'position'):
            a = a.position
        if hasattr(b, 'position'):
            b = b.position

        a = np.asarray(a)
        b = np.asarray(b)
        delta = b - a
        delta, _ = find_mic(delta, cell) 
        return np.linalg.norm(delta)


    @staticmethod
    def _min_distance_to_host_atoms(point, host_atoms, rprimd):
        dists = [BayesianOptimization.min_image_distance(atom, point, rprimd) for atom in host_atoms]
        return np.min(dists)


    @staticmethod
    def _filter_candidates_by_host(X_candidates, host_atoms, rmin, rprimd):
        """Returns filtered ndarray of candidates."""
        mask = [BayesianOptimization._min_distance_to_host_atoms(p, host_atoms, rprimd) > rmin 
                for p in X_candidates]
        return X_candidates[mask]


    @staticmethod
    def _min_distance_within_set(points, new_point, rprimd):
        if len(points) == 0:
            return np.inf
        dists = [BayesianOptimization.min_image_distance(p, new_point, rprimd) for p in points]
        return np.min(dists)


    def _filter_candidates_by_all(self, X_candidates, k):
        """
        Filters candidates that do not fit the conditions:
        - for each atom: distance to host > rmin
        - for each pair in configuration: distance > rmin_insrt (если k > 1)

        Parameters
        ----------
        X_candidates : ndarray of shape (n_candidates, 3*k)
        k : int

        Returns
        -------
        filtered : ndarray
        """
        valid_mask = []
        for candidate_flat in X_candidates:
            points = candidate_flat.reshape(k, 3)
            valid = True

            for point in points:
                d_min = self._min_distance_to_host_atoms(point, self.host, self.rprimd)
                if d_min <= self.rmin:
                    valid = False
                    break

            if not valid:
                valid_mask.append(False)
                continue

            if k > 1:
                for i in range(k):
                    for j in range(i+1, k):
                        dist = BayesianOptimization.min_image_distance(points[i], points[j], self.rprimd)
                        if dist <= self.rmin_insrt:
                            valid = False
                            break
                    if not valid:
                        break

            valid_mask.append(valid)

        return X_candidates[valid_mask]



    def suggest(self, batch_size=1, xi=0.01, strategy='constant_liar'):
        """
        Suggests new configurations for evaluation (with DFT).

        Parameters
        ----------
        batch_size : int, default=1
            The number of configurations to suggest (for parallel computations).

        xi : float, default=0.01
            The exploration/exploitation compromise parameter in the acquisition function.

        strategy : {'constant_liar', 'top_k'}, default='constant_liar'
            A strategy to form batch if batch_size > 1:
            - 'constant_liar' : 
                after choosing configuration predict mean y and add to the temporary training set 
                and re-calculates EI for the rest of the points
            - 'top_k' : 
                just choose top-<batch_size> configurations with the biggest EI.
        Returns
        -------
        X_next : ndarray of shape (batch_size, n_features)
            Suggested configurations (each line is a vector of atomic coordinates).
        """

        n_features = self.X_.shape[1]
        k = n_features // 3 

        if n_features % 3 != 0:
            raise ValueError("The number of features must be multiple of 3 (atomic coordinates x, y, z).")

        X_candidates = self._generate_candidates(self.n_candidates, k)
        X_candidates = self._filter_candidates_by_all(X_candidates, k)

        if len(X_candidates) == 0:
            raise RuntimeError("No acceptable candidates after filtration."
                               "Increase n_candidates or loosen the restrictions (rmin, rmin_insrt).")

        if batch_size == 1:
            ei = self._acquisition(X_candidates, xi=xi)
            best_idx = np.argmax(ei)
            return X_candidates[best_idx].reshape(1, -1)

        # for batch_size > 1

        X_selected = []
        X_temp = self.X_.copy()
        y_temp = self.y_.copy()

        candidates_left = X_candidates.copy()

        for _ in range(batch_size):

            ei = self._acquisition(candidates_left, xi=xi, y=y_temp)

            sorted_indices = np.argsort(ei)[::-1]

            if strategy == 'top_k':
                best_idx = sorted_indices[0]

            elif strategy == 'constant_liar':
                best_idx = sorted_indices[0]

            else:
                raise ValueError("Unknown strategy. Choose 'constant_liar' or 'top_k'.")

            X_best = candidates_left[best_idx].reshape(1, -1)
            X_selected.append(X_best.flatten())


            if strategy == 'constant_liar':
                mu, _ = self._predict(X_best)
                X_temp = np.vstack([X_temp, X_best])
                y_temp = np.append(y_temp, mu)

            candidates_left = np.delete(candidates_left, best_idx, axis=0)
            if len(candidates_left) == 0:
                break

        return np.array(X_selected)




