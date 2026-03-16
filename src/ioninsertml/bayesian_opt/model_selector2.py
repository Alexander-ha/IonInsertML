import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from scipy.linalg import cho_solve, cholesky, solve_triangular

class ModelSelectorMLE:
    """
    Optimizes kernel hyperparameters by maximizing the log marginal likelihood
    (MLE) of a Gaussian process.
    """

    def __init__(self, bo, X_train, y_train, kernel, eta=0.01,
                 normalize_y=True, normalize_X=True):
        self.bo = bo
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.kernel = kernel
        self.eta = eta
        self.normalize_y = normalize_y
        self.normalize_X = normalize_X

        if self.normalize_y:
            self.y_scaler = StandardScaler()
            self.y_train_norm = self.y_scaler.fit_transform(
                self.y_train.reshape(-1, 1)).ravel()
        else:
            self.y_scaler = None
            self.y_train_norm = self.y_train

        if self.normalize_X:
            self.scaler_X = StandardScaler()
            self.X_train = self.scaler_X.fit_transform(self.X_train)
        else:
            self.scaler_X = None

        self.kernel_opt_ = None
        self.n_samples = len(self.X_train)

    def _marginal_ll_and_grad(self, theta):
        """
        Compute negative log marginal likelihood and its gradient
        with better numerical stability.
        """
        # Клонируем ядро с новыми гиперпараметрами
        kernel_t = self.kernel.clone_with_theta(theta)
        
        # Вычисляем матрицу ковариации и ее градиент
        K, K_grad = kernel_t(self.X_train, eval_gradient=True)
        
        # Добавляем jitter для численной стабильности
        jitter = 1e-6 * np.mean(np.diag(K))
        K_reg = K + jitter * np.eye(self.n_samples)
        
        try:
            # Используем cholesky для стабильности
            L = cholesky(K_reg, lower=True)
        except np.linalg.LinAlgError:
            # Если все еще нестабильно, увеличиваем jitter
            jitter = 1e-4 * np.mean(np.diag(K))
            K_reg = K + jitter * np.eye(self.n_samples)
            L = cholesky(K_reg, lower=True)
        
        # Решаем систему K * alpha = y
        alpha = cho_solve((L, True), self.y_train_norm)
        
        # Вычисляем log детерминант через разложение Холецкого
        log_det = 2 * np.sum(np.log(np.diag(L)))
        
        # Вычисляем log marginal likelihood
        n = self.n_samples
        log_likelihood = -0.5 * np.dot(self.y_train_norm, alpha) - \
                         0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
        
        # Вычисляем градиент
        # K_inv = L^-T * L^-1
        K_inv = cho_solve((L, True), np.eye(n))
        
        # alpha * alpha^T - K_inv
        alpha_alphaT = np.outer(alpha, alpha)
        term = alpha_alphaT - K_inv
        
        n_params = len(theta)
        grad = np.zeros(n_params)
        
        for j in range(n_params):
            # Более стабильное вычисление следа
            grad_j = 0.5 * np.sum(term * K_grad[:, :, j])
            grad[j] = grad_j
        
        return -log_likelihood, -grad

    def find_via_scipy(self, method='L-BFGS-B', options=None, n_restarts=5):
        """
        Optimize with multiple restarts to avoid local minima.
        """
        bounds = self.kernel.bounds
        best_nll = np.inf
        best_theta = None
        
        # Пробуем разные начальные точки
        for i in range(n_restarts):
            if i == 0:
                # Начинаем с текущих значений
                x0 = self.kernel.theta
            else:
                # Случайные начальные точки в лог-пространстве
                x0 = np.array([
                    np.random.uniform(
                        np.log(bound[0]) if bound[0] > 0 else bound[0],
                        np.log(bound[1]) if bound[1] > 0 else bound[1]
                    ) if bound[0] > 0 else np.random.uniform(bound[0], bound[1])
                    for bound in bounds
                ])
            
            res = minimize(
                fun=lambda th: self._marginal_ll_and_grad(th)[0],
                jac=lambda th: self._marginal_ll_and_grad(th)[1],
                x0=x0,
                method=method,
                bounds=bounds,
                options=options or {'maxiter': 200, 'disp': False}
            )
            
            if res.success and res.fun < best_nll:
                best_nll = res.fun
                best_theta = res.x
        
        if best_theta is None:
            # Если ничего не сработало, используем исходные параметры
            print("Warning: Optimization failed, using original kernel parameters")
            best_theta = self.kernel.theta
        
        self.kernel_opt_ = self.kernel.clone_with_theta(best_theta)
        return self.kernel_opt_

    def transform_y(self, y):
        """Apply target normalization if fitted."""
        if self.y_scaler is not None:
            return self.y_scaler.transform(np.asarray(y).reshape(-1, 1)).ravel()
        return y

    def inverse_transform_y(self, y_norm):
        """Revert target normalization if fitted."""
        if self.y_scaler is not None:
            return self.y_scaler.inverse_transform(np.asarray(y_norm).reshape(-1, 1)).ravel()
        return y_norm