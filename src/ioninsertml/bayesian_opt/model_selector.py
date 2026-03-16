import numpy as np
import pandas as pd

from ase.geometry import find_mic
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

class ModelSelector:
    '''
    Performs LOO-CV hyperparams obtatining.
    Parameters:
    -----------
    optimizer: 'BayesianOptimization' class preferable.
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    kernel: sklearn.gaussian_process.kernels to be fitted.
    '''

    def __init__(self, bo, X_train, y_train, kernel, eta=0.01, normalize_y=True, normalize_X = True):
        self.bo = bo
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.kernel = kernel
        self.eta = eta
        self.normalize_y = normalize_y
        self.normalize_X = normalize_X
        if self.normalize_y:
            self.y_scaler = StandardScaler()
            self.y_train_norm = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).ravel()
        else:
            self.y_scaler = None
            self.y_train_norm = self.y_train
        if self.normalize_X:
            self.scaler_X = StandardScaler()
            self.X_train = self.scaler_X.fit_transform(X_train)
        else:
            self.scaler_X = None
            self.X_train = self.X_train


        self.kernel_opt = None
    
    def LOO_Compute(self, theta):
        '''
        this func computes LOO-CV
        '''
        kernel_t = self.kernel.clone_with_theta(theta) #вот тут вытаскиваем гиперпараметры ядра

        #потом идея как в статье (которую скину в чат)
        K, K_grad = kernel_t(self.X_train, eval_gradient=True)
        noise = 1e-8 * np.trace(K) / len(K)
        K_reg = K + noise * np.eye(len(K))#для численной стабильности на случай если у нас есть линейно зависимые (коррелирующие сэмплы)
        K_inv = np.linalg.inv(K_reg)
        alpha = K_inv @ self.y_train_norm #альфа из статьи
        diag_K_inv = np.diag(K_inv)
        mu_loo = self.y_train_norm - alpha/diag_K_inv #mean
        s2_loo = 1.0 / diag_K_inv #standart derivation
        logp = -0.5 * np.log(s2_loo) - 0.5 * (self.y_train_norm-mu_loo)**2 / s2_loo - 0.5 * np.log(2*np.pi)
        loo = np.sum(logp) #loo по всем вариантам
        n_params = len(theta)
        grad = np.zeros(n_params)
        for j in range(n_params):
            Z_j = K_inv@K_grad[:, :, j]
            Z_j_K_inv = Z_j @ K_inv
            Z_a = Z_j @ alpha
            contrib = (alpha * Z_a - 0.5 * (1.0 + alpha**2 / diag_K_inv)*np.diag(Z_j_K_inv))/diag_K_inv
            grad[j] = np.sum(contrib)
        return loo, grad

    def get_looNgrad(self, theta):
        loo, grad = self.LOO_Compute(theta)
        return -loo, -grad

    def find_via_scipy(self, method='L-BFGS-B', options=None):
        """
        Optimize hyperparameters using scipy.optimize.minimize.
        """
        x0 = self.kernel.theta
        bounds = self.kernel.bounds

        res = minimize( fun=lambda th: self.get_looNgrad(th)[0],jac=lambda th: self.get_looNgrad(th)[1],x0=x0, method=method, bounds=bounds,options=options or {'maxiter': 200, 'disp': False}
        )

        if res.success:
            opt_theta = res.x
        else:
            print("failed for scipy optimization, keeping kernela s it was...")
            opt_theta = x0
        self.kernel_opt = self.kernel.clone_with_theta(opt_theta)
        return self.kernel_opt

    def find_via_gd(self, n_iter=100, tol=1e-5, verbose=False):
        theta = self.kernel.theta.copy()
        prev_loo = -np.inf

        for i in range(n_iter):
            loo, grad = self._compute_loo_and_grad(theta)
            if verbose:
                print(f"Iter {i}: LOO = {loo:.4f}")

            if loo < prev_loo + tol:
                break
            prev_loo = loo
            theta += self.eta * grad
            bounds = self.kernel.bounds
            for j, (lb, ub) in enumerate(bounds):
                if lb is not None:
                    theta[j] = max(lb, theta[j])
                if ub is not None:
                    theta[j] = min(ub, theta[j])
        self.kernel_opt_ = self.kernel.clone_with_theta(theta)
        self.bo.gpr_.optimizer = None
        return self.kernel_opt_

    def fit(self, method='scipy', **kwargs):
        """
        Main entry point to optimize hyperparameters.

        Parameters:
        -----------
        method : str, either 'scipy' or 'gd'
            Optimization method.
        **kwargs : additional arguments passed to the specific optimizer.
        """
        if method.lower() == 'scipy':
            return self.optimize_hyperparameters_scipy(**kwargs)
        elif method.lower() == 'gd':
            return self.optimize_hyperparameters_gd(**kwargs)
        else:
            raise ValueError("method must be 'scipy' or 'gd'")

    def get_optimized_kernel(self):
        if self.kernel_opt_ is None:
            raise RuntimeError("No optimized kernel available. Call fit() first.")
        return self.kernel_opt_

    def transform_y(self, y):
        if self.y_scaler is not None:
            return self.y_scaler.transform(np.asarray(y).reshape(-1, 1)).ravel()
        return y

    def inverse_transform_y(self, y_norm):
        if self.y_scaler is not None:
            return self.y_scaler.inverse_transform(np.asarray(y_norm).reshape(-1, 1)).ravel()
        return y_norm



