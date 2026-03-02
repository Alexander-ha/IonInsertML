import numpy as np
from siman.geo import image_distance

def generate_config(host_atoms, rprimd, n_candidates=100, rmin=1.45, n_configs=25):

    """
    A function to generate a random position for 1 inserted atom

    INPUT:
        - host_atoms (np.array): an array of cartesian coordinates of the host
                                 (siman) can be extracted as Structure().xcart[i:j]

        - rprimd (np.array): an array with cell vectors
                                 (siman) can be extracted as Structure().rprimd

        - n_candidates (int): a number of random configurations to generate with np.random.uniform
                                from this number of configurations valid ones will be selected

        - rmin (float): a minimum distance possible between inserted atom and host atoms types
                        for Li in graphite matrix a sum of Li ionic radius and C atomic radius is used as rmin
                        
        - n_configs (int): the number of configurations you want to get
                           (from randomly sampled and filtered n_candidates)

    RETURNS
        - a list of cartesian coordinates of len n_configs for atom to insert in given host
    """
    
    frac_coordinates = np.random.uniform(
    low=0.0,
    high=1.0,
    size=(n_candidates, 3)
    )

    if type(type) == list:
        rprimd = np.array(rprimd)
    
    X_candidates = frac_coordinates @ rprimd

    def min_distance_to_host_atoms(point, host_atoms):
        """Defines a mask to filter generated coordinates of inserted atoms with respect to host atoms positions"""
        dists = []
        for atom in host_atoms:
            dist = image_distance(atom, point, rprimd)
            dists.append(dist)
        return np.min(dists)

    def filter_candidates(X_candidates, host_atoms, rmin):
        """The filter with respect to minimum distance to host atoms"""
        mask = [min_distance_to_host_atoms(p, host_atoms) > rmin for p in X_candidates]
        return X_candidates[mask]
    
    X_candidates = filter_candidates(X_candidates, host_atoms, rmin)
    if len(X_candidates) == 0:
        print("No suitable candidates. Increase the area or decrease rmin.")
        return []

    if X_candidates.shape[0] > n_configs:
        rng = np.random.default_rng() 
        X_configs = rng.choice(X_candidates, size=n_configs, replace=False)
        
    return X_configs



def generate_n_configs(host_atoms, rprimd, n_insrt, n_candidates=100, rmin=1.45, rmin_insrt=1.34, n_configs=25):

        """
    A function to generate a random position for n inserted atoms

    INPUT:
        - host_atoms (np.array): an array of cartesian coordinates of the host
                                 (siman) can be extracted as Structure().xcart[i:j]

        - rprimd (np.array): an array with cell vectors
                                 (siman) can be extracted as Structure().rprimd

        - n_insrt (int): the number of atoms to be inserted per host matrix

        - n_candidates (int): a number of random configurations to generate with np.random.uniform
                                from this number of configurations valid ones will be selected

        - rmin (float): a minimum distance possible between inserted atom and host atoms types
                        for Li in graphite matrix a sum of Li ionic radius and C atomic radius is used as rmin

        - rmin_insrt (float): a minimum distance possible between inserted atoms
                              for Li in graphite matrix a doubled Li ionic radius is used as rmin_insrt

        - n_configs (int): the number of configurations you want to get
                           (from randomly sampled and filtered n_candidates)

    RETURNS
        - a list of cartesian coordinates of len n_configs for atom to insert in given host
    """

    if rmin_insrt == None:
        rmin_insrt = rmin

    frac_coordinates = np.random.uniform(
    low=0.0,
    high=1.0,
    size=(n_candidates, 3)
    )

    if type(type) == list:
        rprimd = np.array(rprimd)
    
    X_candidates = frac_coordinates @ rprimd

    def min_distance_to_host_atoms(point, host_atoms):
        """Defines a mask to filter generated coordinates of inserted atoms with respect to host atoms positions"""
        dists = []
        for atom in host_atoms:
            dist = image_distance(atom, point, rprimd)
            dists.append(dist)
        return np.min(dists)

    def filter_candidates(X_candidates, host_atoms, rmin):
        """The filter with respect to minimum distance to host atoms"""
        mask = [min_distance_to_host_atoms(p, host_atoms) > rmin for p in X_candidates]
        return X_candidates[mask]

    def min_distance_within_set(points, new_point, rprimd):
        """Defines a mask to filter generated coordinates of inserted atoms with respect to generated inserted atoms positions"""
        if len(points) == 0:
            return np.inf
        dists = [image_distance(p, new_point, rprimd) for p in points]
        return np.min(dists)
    
    X_candidates = filter_candidates(X_candidates, host_atoms, rmin)
    
    if len(X_candidates) == 0:
        print("No suitable candidates. Increase the area or decrease rmin.")
        return []

    X_configs = []
    max_attempts_per_config = 50

    pool = X_candidates.copy()
    
    for _ in range(n_configs):
        found = False
        for attempt in range(max_attempts_per_config):
            np.random.shuffle(pool)
            selected = []
            for point in pool:
                if min_distance_within_set(selected, point, rprimd) > rmin_insrt:
                    selected.append(point)
                    if len(selected) == n_insrt:
                        break
            if len(selected) == n_insrt:
                X_configs.append(np.array(selected))
                found = True
                break
        if not found:
            print(f"Couldn't build configuration with {n_insrt} atoms after {max_attempts_per_config} attempts.")
            continue
    
    return X_configs


