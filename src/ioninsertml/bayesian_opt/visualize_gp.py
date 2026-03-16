#!/usr/bin/env python3
"""
Visualization of Gaussian Process predictions with optimized hyperparameters.
This script loads the training data and the optimized kernel from a previous
Bayesian optimization run, refits the GP, and produces plots to assess the
interpolation behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel,ConstantKernel
from sklearn.preprocessing import StandardScaler
import os
import argparse

def load_data(data_dir, iteration=5):
    """
    Load training data from numpy files saved during BO run.
    """
    X_file = os.path.join(data_dir, f'X_train_iter_{iteration}.npy')
    y_file = os.path.join(data_dir, f'y_train_iter_{iteration}.npy')
    
    if not os.path.exists(X_file) or not os.path.exists(y_file):
        raise FileNotFoundError(f"Data files not found in {data_dir}")
    
    X = np.load(X_file)
    y = np.load(y_file)
    return X, y

def plot_1d_slice(X, y, gp, scaler_X, scaler_y,
                  fixed_coords=None, slice_dim=0,
                  n_points=200, confidence=2.0,
                  save_path=None, show_plot=False): 
    """
    Plot a 1D slice of the GP prediction along a given dimension.
    
    Parameters
    ----------
    show_plot : bool, default=False
        If True, call plt.show() to display the plot interactively.
        If False, only save the figure if save_path is provided.
    """
    n_features = X.shape[1]
    if fixed_coords is None:
        fixed_coords = np.median(X, axis=0)
    else:
        fixed_coords = np.asarray(fixed_coords)
        assert fixed_coords.shape == (n_features,), "fixed_coords must match feature dimension"
    
    # Create grid along slice_dim
    x_min, x_max = X[:, slice_dim].min(), X[:, slice_dim].max()
    # Extend a bit beyond training range to see extrapolation
    margin = 0.1 * (x_max - x_min)
    x_vals = np.linspace(x_min - margin, x_max + margin, n_points)
    
    # Build input matrix: fixed values for all dims, varying only slice_dim
    X_grid = np.tile(fixed_coords, (n_points, 1))
    X_grid[:, slice_dim] = x_vals
    
    # Scale and predict
    X_grid_scaled = scaler_X.transform(X_grid)
    y_mean_scaled, y_std_scaled = gp.predict(X_grid_scaled, return_std=True)
    y_mean = scaler_y.inverse_transform(y_mean_scaled.reshape(-1, 1)).ravel()
    y_std = y_std_scaled * scaler_y.scale_
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_mean, 'b-', label='GP mean')
    plt.fill_between(x_vals,
                     y_mean - confidence * y_std,
                     y_mean + confidence * y_std,
                     alpha=0.3, color='b', label=f'{confidence}σ interval')
    plt.scatter(X[:, slice_dim], y, c='r', marker='o', label='Training data')
    plt.xlabel(f'Feature {slice_dim} (original scale)')
    plt.ylabel('Energy (eV)')
    plt.title(f'GP slice along dimension {slice_dim} (other features fixed at median)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close() 

def plot_covariance_matrix(gp, scaler_X, X, save_path=None, show_plot=False):
    """
    Visualize the covariance matrix of the training points (scaled space).
    """
    X_scaled = scaler_X.transform(X)
    K = gp.kernel_(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.imshow(K, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Covariance')
    plt.title('Covariance matrix of training points (scaled)')
    plt.xlabel('Point index')
    plt.ylabel('Point index')
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Covariance matrix saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize GP predictions')
    parser.add_argument('--data_dir', type=str, default='bo_test_results',
                        help='Directory containing saved numpy arrays')
    parser.add_argument('--iteration', type=int, default=5,
                        help='Iteration number of the saved data (default: 5)')
    parser.add_argument('--length_scale', type=float, default=1e-5,
                        help='RBF length scale (from optimization)')
    parser.add_argument('--noise_level', type=float, default=1.5e-5,
                        help='White kernel noise level (from optimization)')
    parser.add_argument('--slice_dim', type=int, default=0,
                        help='Dimension to vary in 1D slice')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save figures (if not specified, plots are shown interactively)')
    parser.add_argument('--save_fig', type=str, default=None,
                        help='Explicit filename for slice plot (overrides automatic naming)')
    parser.add_argument('--save_cov', type=str, default=None,
                        help='Explicit filename for covariance matrix plot (overrides automatic naming)')
    parser.add_argument('--const_coef', type=float, default=1.06**2,
                        help='Explicit filename for covariance matrix plot (overrides automatic naming)')
    args = parser.parse_args()
    
    # Load data
    X, y = load_data(args.data_dir, args.iteration)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Energy range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Prepare scalers (as done in BayesianOptimization.fit)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Create kernel with given hyperparameters
    rbf =  ConstantKernel(args.const_coef) * RBF(length_scale=args.length_scale) + WhiteKernel(noise_level=args.noise_level)
    gp = GaussianProcessRegressor(kernel=rbf, optimizer=None, normalize_y=False)
    gp.fit(X_scaled, y_scaled)
    print("GP refitted with optimized kernel.")
    print(f"Final kernel: {gp.kernel_}")
    
    # Determine output paths
    slice_path = args.save_fig
    cov_path = args.save_cov
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        if not slice_path:
            slice_path = os.path.join(args.output_dir, f'slice_dim{args.slice_dim}_ls{args.length_scale}.png')
        if not cov_path:
            cov_path = os.path.join(args.output_dir, 'covariance_matrix.png')
    
    # Plot 1D slice
    plot_1d_slice(X, y, gp, scaler_X, scaler_y,
                slice_dim=args.slice_dim,
                save_path=slice_path,
                show_plot=False)

    # Plot covariance matrix
    plot_covariance_matrix(gp, scaler_X, X, 
                        save_path=cov_path,
                        show_plot=False)

    print(f"Plots saved to {args.output_dir if args.output_dir else 'specified files'}")
    # Plot covariance matrix
    if cov_path or (args.output_dir is None and cov_path is None):
        # If no output_dir and no explicit cov path, still show interactively
        plot_covariance_matrix(gp, scaler_X, X, save_path=cov_path)
    
    # If neither output_dir nor explicit files, keep plots open (already handled by interactive show)

if __name__ == "__main__":
    main()