import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

class ModelSelectorMOD:
    '''
    Performs LOO-CV hyperparameters optimization.
    
    Parameters:
    -----------
    bo: BayesianOptimization instance
    X_train: ndarray of shape (n_samples, n_features)
    y_train: ndarray of shape (n_samples,)
    kernel: sklearn.gaussian_process.kernels to be fitted.
    eta: float, learning rate for gradient descent (unused if using scipy)
    normalize_y: bool, whether to normalize target values
    normalize_X: bool, whether to normalize features
    '''

    def __init__(self, bo, X_train, y_train, kernel, eta=0.01, normalize_y=True, normalize_X=True):
        self.bo = bo
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.kernel = kernel
        self.eta = eta
        self.normalize_y = normalize_y
        self.normalize_X = normalize_X
        
        if self.normalize_X:
            self.scaler_X = StandardScaler()
            self.X_train_norm = self.scaler_X.fit_transform(self.X_train)
        else:
            self.scaler_X = None
            self.X_train_norm = self.X_train
            
        if self.normalize_y:
            self.y_scaler = StandardScaler()
            self.y_train_norm = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).ravel()
        else:
            self.y_scaler = None
            self.y_train_norm = self.y_train

        self.kernel_opt = None
    
    def LOO_Compute(self, theta):
        '''
        Compute LOO-CV log probability and its gradient.
        
        Parameters:
        -----------
        theta: array, kernel hyperparameters
        
        Returns:
        --------
        loo: float, LOO log probability
        grad: array, gradient of LOO w.r.t. hyperparameters
        '''
        kernel_t = self.kernel.clone_with_theta(theta)
        
        K, K_grad = kernel_t(self.X_train_norm, eval_gradient=True)
        
        n = len(K)
        jitter = 1e-8 * np.trace(K) / n
        K_reg = K + jitter * np.eye(n)
        L, lower = cho_factor(K_reg, lower=True)
        K_inv = cho_solve((L, lower), np.eye(n))
        alpha = cho_solve((L, lower), self.y_train_norm)

        diag_K_inv = np.diag(K_inv)
        
        mu_loo = self.y_train_norm - alpha / diag_K_inv
        s2_loo = 1.0 / diag_K_inv
        
        logp = -0.5 * np.log(s2_loo) - 0.5 * (self.y_train_norm - mu_loo)**2 / s2_loo - 0.5 * np.log(2 * np.pi)
        loo = np.sum(logp)
        
        n_params = len(theta)
        grad = np.zeros(n_params)
        
        for j in range(n_params):
            Z_j = K_inv @ K_grad[:, :, j]
            Z_j_K_inv = Z_j @ K_inv
            Z_alpha = Z_j @ alpha
            contrib = (alpha * Z_alpha - 0.5 * (1.0 + alpha**2 / diag_K_inv) * np.diag(Z_j_K_inv)) / diag_K_inv
            grad[j] = np.sum(contrib)
            
        return loo, grad

    def get_looNgrad(self, theta):
        '''Wrapper for minimization (returns negative LOO)'''
        loo, grad = self.LOO_Compute(theta)
        return -loo, -grad

    def find_via_scipy(self, method='L-BFGS-B', options=None, n_restarts=1):
        x0 = self.kernel.theta
        bounds = self.kernel.bounds
        best_res = None
        best_loo = -np.inf

        for _ in range(n_restarts):
            if _ > 0:
                x0 = np.array([np.random.uniform(low, high) for low, high in bounds])
            res = minimize(
                fun=lambda th: self.get_looNgrad(th)[0],
                jac=lambda th: self.get_looNgrad(th)[1],
                x0=x0,
                method=method,
                bounds=bounds,
                options=options
            )
            if res.success and -res.fun > best_loo:
                best_loo = -res.fun
                best_res = res
        if best_res is None:
            best_res = res 
        opt_theta = best_res.x
        self.kernel_opt = self.kernel.clone_with_theta(opt_theta)
        return self.kernel_opt

    def transform_X(self, X):
        """Transform features using fitted scaler"""
        if self.scaler_X is not None:
            return self.scaler_X.transform(X)
        return X

    def transform_y(self, y):
        """Transform targets using fitted scaler"""
        if self.y_scaler is not None:
            return self.y_scaler.transform(np.asarray(y).reshape(-1, 1)).ravel()
        return y

    def inverse_transform_y(self, y_norm):
        """Inverse transform normalized targets"""
        if self.y_scaler is not None:
            return self.y_scaler.inverse_transform(np.asarray(y_norm).reshape(-1, 1)).ravel()
        return y_norm