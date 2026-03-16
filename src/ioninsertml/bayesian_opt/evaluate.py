import numpy as np

import pandas as pd
import os
import sys
import argparse
import logging

from ase import Atom
from ase import Atoms
from ase.io import read

from ioninsertml.bayesian_opt.model_selector import ModelSelector
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from ioninsertml.bayesian_opt.bo import BayesianOptimization
from ioninsertml.utils.data_loader import load_train, load_host
from ioninsertml.bayesian_opt.model_selector1 import ModelSelectorMOD
from ioninsertml.bayesian_opt.model_selector2 import ModelSelectorMLE


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class DummyVasp:
    
    def __init__(self, directory=None, **kwargs):
        self.directory = directory
        self.params = kwargs
        self.rng = np.random.RandomState(42)
        
    def get_potential_energy(self, atoms=None):
        if self.directory:
            poscar_file = os.path.join(self.directory, 'POSCAR')
            with open(poscar_file, 'w') as f:
                f.write("Test POSCAR file\n")
        
        base_energy = -10.0
        
        if atoms and len(atoms) > 0:
            li_pos = atoms[-1].position
            center = np.array([2.5, 2.5, 5.0])
            dist_to_center = np.linalg.norm(li_pos - center)
            position_effect = -0.5 * np.exp(-dist_to_center / 3.0)
            noise = self.rng.normal(0, 0.1)
            
            total_energy = base_energy + position_effect + noise
        else:
            total_energy = base_energy + self.rng.uniform(-0.5, 0.5)
            
        return total_energy

def read_incar(incar_file):
    params = {}
    try:
        with open(incar_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().split('#')[0].strip()
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                    params[key] = value
                    
        logging.info(f"Successfully read {len(params)} parameters from INCAR")
        return params
    except Exception as e:
        logging.error(f"Error reading INCAR file: {e}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Bayesian Optimization for Ion Insertion (TEST MODE)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'train_file_csv',
        type=str,
        help='Path to train CSV'
    )
    
    parser.add_argument(
        'host_file',
        type=str,
        help='Path to POSCAR'
    )
    
    parser.add_argument(
        'params_file',
        type=str,
        help='Path to INCAR parameters file'
    )
    
    parser.add_argument(
        'host_energy',
        type=float,
        help='Host energy for y_new calcs'
    )
    
    parser.add_argument(
        'int_atom_energy',
        type=float,
        help='Reference energy for y_new calcs'
    )
    
    parser.add_argument(
        '--int_atom',
        type=str,
        default='Li',
        help='Type of atom inserted in the host structure'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=3,
        help='Size of points (simultaneos) to calc'
    )
    
    parser.add_argument(
        '--strat',
        type=str,
        default='constant_liar',
        choices=['constant_liar', 'cl_min', 'cl_max', 'cl_mean'],
        help='Batch formation strategy'
    )
    
    parser.add_argument(
        '--N',
        type=int,
        default=5,
        help='Number of active learning iters.'
    )
    
    parser.add_argument(
        '--n_candidates',
        type=int,
        default=500,
        help='Candidates num for BO'
    )
    
    parser.add_argument(
        '--rmin',
        type=float,
        default=1.45,
        help='Minimum distance between hosted C atoms'
    )
    
    parser.add_argument(
        '--rmin_insrt',
        type=float,
        default=1.34,
        help='Minimum distance between inserted atoms'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    parser.add_argument(
        '--xi',
        type=float,
        default=0.01,
        help='Exploration parameter for acquisition function'
    )
    
    parser.add_argument(
        '--test_mode',
        action='store_true',
        default=True,
        help='Use dummy calculator for testing'
    )
    
    parser.add_argument(
        '--loo_frequency',
        type=int,
        default=5,
        help='Frequency of LOO hyperparameter optimization (0 = only at start)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Number of iterations without improvement to stop'
    )
    
    parser.add_argument(
        '--length_scale_bounds',
        type=float,
        nargs=2,
        default=[0.01, 100.0],
        help='Bounds for RBF length scale'
    )
    
    return parser.parse_args()

def compute_y(total_energy, host_energy, int_atom_energy, n_int_atoms):
    """intercalation energy computation for a new configuration suggested by BO"""
    return (total_energy - host_energy - int_atom_energy) / n_int_atoms

def save_results(X_train, y_train, iteration, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, f'X_train_iter_{iteration}.npy'), X_train)
    np.save(os.path.join(output_dir, f'y_train_iter_{iteration}.npy'), y_train)
    
    df = pd.DataFrame(X_train)
    df['energy'] = y_train
    df.to_csv(os.path.join(output_dir, f'training_data_iter_{iteration}.csv'), index=False)
    
    logging.info(f"Results saved to {output_dir}")

def check_improvement(y_train, best_energy_so_far, tolerance=1e-4):
    """Check if there's been improvement in the best energy"""
    current_best = y_train.min()
    improvement = best_energy_so_far - current_best
    return current_best, improvement > tolerance

def main():
    print("Configuring logging...")
    """Main function with command line arguments"""
    args = parse_arguments()
    logging.info("BO Test Mode Parameters:")
    for key, value in vars(args).items():
        logging.info(f" {key}: {value}")

    try:
        logging.info(f"Reading VASP parameters from {args.params_file}")
        params = read_incar(args.params_file)

        logging.info(f"Loading host structure from {args.host_file}")
        host_atoms = load_host(args.host_file)
        host_positions = host_atoms.get_positions()
        rprimd = host_atoms.get_cell()[:]

        logging.info(f"Loading training data from {args.train_file_csv}")
        X_init, y_init = load_train(args.train_file_csv)
        logging.info(f"Loaded {len(X_init)} training points")
        logging.info(f"Energy range: [{y_init.min():.3f}, {y_init.max():.3f}]")

        logging.info("Initializing Bayesian Optimization")

        kernel = ConstantKernel(1.0, constant_value_bounds=(0.01, 10.0)) * RBF(
            length_scale=1.0,
            length_scale_bounds=(0.1, 10.0)
        ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1.0))
        bo = BayesianOptimization(
            int_atom=args.int_atom,
            n_candidates=args.n_candidates,
            rmin=args.rmin,
            rmin_insrt=args.rmin_insrt,
            host=host_positions,
            rprimd=rprimd,
            random_state=args.random_state,
            kernel=kernel
        )

        X_train, y_train = X_init.copy(), y_init.copy()

        logging.info("Optimizing kernel hyperparameters via LOO-CV (once at start)")

        bo.fit(X_train, y_train)

        selector = ModelSelectorMLE(
            bo=bo,
            X_train=X_train, 
            y_train=y_train,  
            kernel=bo.kernel,
            normalize_y=True, 
            normalize_X=True   
        )

        optimized_kernel = selector.find_via_scipy(
            method='L-BFGS-B',
            n_restarts=10,
            options={'maxiter': 500, 'ftol': 1e-6}
        )

        bo.gpr_.kernel = optimized_kernel
        bo.kernel = optimized_kernel
        bo.gpr_.optimizer = None
        bo.fit(X_train, y_train)

        logging.info(f"Final kernel after re-fit: {bo.gpr_.kernel}")

        best_energy = y_train.min()
        no_improvement_count = 0
        results_dir = 'bo_test_results'
        os.makedirs(results_dir, exist_ok=True)
        save_results(X_train, y_train, 0, results_dir)

        iteration = 0
        while iteration < args.N:
            logging.info(f"Iteration {iteration + 1}/{args.N}")
            logging.info(f"Current training set size: {len(X_train)}")
            if args.loo_frequency > 0 and iteration > 0 and iteration % args.loo_frequency == 0:
                logging.info(f"Periodic LOO hyperparameter optimization (iteration {iteration})")

                X_train_scaled = bo.scaler_X.transform(X_train)
                y_train_scaled = bo.scaler_y.transform(y_train.reshape(-1, 1)).ravel()

                selector = ModelSelectorMLE(
                    bo=bo,
                    X_train=X_train_scaled,
                    y_train=y_train_scaled,
                    kernel=bo.kernel,
                    normalize_y=False,
                    normalize_X=False
                )

                new_kernel = selector.find_via_scipy(method='L-BFGS-B')
                logging.info(f"Re-optimized kernel: {new_kernel}")
                logging.info(f"New length_scale: {new_kernel.k1.k2.length_scale:.4f}")

                bo.gpr_.kernel = new_kernel
                bo.kernel = new_kernel
                bo.gpr_.optimizer = None
                bo.fit(X_train, y_train)  # Refit GP

                logging.info(f"Updated kernel: {bo.gpr_.kernel}")

            logging.info(f"Using kernel (fixed): {bo.gpr_.kernel}")
            X_new = bo.suggest(batch_size=args.batch_size, xi=args.xi, strategy=args.strat)
            logging.info(f"Suggested {len(X_new)} new configurations")

            for i, position in enumerate(X_new):
                atoms = host_atoms.copy()
                atoms.append(Atom(args.int_atom, position=position))
                calc_dir = f'calc_{iteration}_{i}'
                os.makedirs(calc_dir, exist_ok=True)
                calc = DummyVasp(directory=calc_dir, **params)
                atoms.calc = calc

                logging.info(f"Running dummy calculation in {calc_dir}")
                try:
                    total_energy = atoms.get_potential_energy()
                    logging.info(f"Dummy calculation successful, energy: {total_energy:.6f} eV")
                except Exception as e:
                    logging.error(f"Dummy calculation failed for {calc_dir}: {e}")
                    continue

                y_new = compute_y(total_energy, args.host_energy, args.int_atom_energy, n_int_atoms=1)
                X_train = np.vstack([X_train, position.reshape(1, -1)])
                y_train = np.append(y_train, y_new)

                logging.info(f"New intercalation energy: {y_new:.6f} eV")
                poscar_file = os.path.join(calc_dir, 'POSCAR_with_Li')
                atoms.write(poscar_file, format='vasp')
                logging.info(f"Structure saved to {poscar_file}")

            save_results(X_train, y_train, iteration + 1, results_dir)

            new_best_energy, improved = check_improvement(y_train, best_energy)
            #if improved:
            #    logging.info(f"New best energy found: {new_best_energy:.6f} eV (previous: {best_energy:.6f} eV)")
            #    best_energy = new_best_energy
            #    no_improvement_count = 0
            #else:
            #    no_improvement_count += 1
            #    logging.info(f"No improvement in best energy for {no_improvement_count} iterations")

            logging.info(f"Iteration {iteration + 1} completed. Training set size: {len(X_train)}")
            logging.info(f"Energy stats - min: {y_train.min():.3f}, max: {y_train.max():.3f}, mean: {y_train.mean():.3f}")

            if args.patience > 0 and no_improvement_count >= args.patience:
                logging.info(f"Stopping early after {args.patience} iterations without improvement")
                break

            iteration += 1

        logging.info("Optimization completed successfully!")
        logging.info(f"Final training set size: {len(X_train)}")
        logging.info(f"Best energy found: {y_train.min():.6f} eV")

        final_df = pd.DataFrame({
            'iteration': range(len(y_train)),
            'energy': y_train
        })
        final_df.to_csv(os.path.join(results_dir, 'final_results.csv'), index=False)

    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
