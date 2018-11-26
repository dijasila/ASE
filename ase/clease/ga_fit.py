from __future__ import print_function
from ase.clease import Tikhonov
import numpy as np
import os
import multiprocessing as mp
import os
os.environ["OPENBLAS_MAIN_FREE"] = "1"


class GAFit(object):
    """
    Genetic Algorithm for selecting relevant clusters

    Arguments:
    =========
    setting: ClusterExpansionSetting
        Setting object used for the cluster expanion

    max_cluster_dia: float
        Maximum diameter included in the population

    max_cluster_size: int
        Maximum number of atoms included in the largest cluster

    alpha: float
        Regularization parameter for ridge regression which is used internally
        to obtain the coefficient

    elitism: int
        Number of best structures that will be passed unaltered on to the next
        generation

    fname: str
        Filename used to backup the population. If this file exists, the next
        run will load the population from the file and start from there.

    num_individuals: int or str
        Integer with the number of inidivuals or it is equal to "auto",
        in which case 10 times the number of candidate clusters is used

    change_prob: float
        If a mutation is selected this denotes the probability of a mutating
        a given gene.

    max_num_in_init_pool: int
        If given the maximum clusters included in the initial population
        is given by this number. If max_num_in_init_pool=150, then
        solution with maximum 150 will be present in the initial pool.

    parallel: bool
        If True multiprocessing will be used to parallelize over the
        individuals in the population.
        NOTE: One of the most CPU intensive tasks involves matrix
        manipulations using Numpy. If your Numpy installation uses
        hyperthreading, it is possible that running with parallel=True
        actually leads to lower performance.

    num_core: int
        Number of cores to use during parallelization. 
        If not given (and parallel=True) then mp.cpu_count()/2
        will be used
    
    Example:
    =======
    from ase.clease import Evaluate
    from ase.clease import GAFit
    setting = None # Should be an ASE ClusterExpansionSetting object
    evaluator = Evaluate(setting)
    ga_fit = GAFit(evaluator)
    ga_fit.run()
    evaluator.get_cluster_name_eci()

    """
    def __init__(self, setting=None, max_cluster_size=None,
                 max_cluster_dia=None, mutation_prob=0.001, alpha=1E-5,
                 elitism=3, fname="ga_fit.csv", num_individuals="auto",
                 change_prob=0.2, local_decline=True,
                 max_num_in_init_pool=None, parallel=False, num_core=None):
        from ase.clease import Evaluate
        evaluator = Evaluate(setting, max_cluster_dia=max_cluster_dia,
                             max_cluster_size=max_cluster_size)

        # Read required attributes from evaluate
        self.cf_matrix = evaluator.cf_matrix
        self.cluster_names = evaluator.cluster_names
        self.e_dft = evaluator.e_dft
        self.fname = fname

        self.fname_cluster_names = fname.rpartition(".")[0] + "_cluster_names.txt"
        if num_individuals == "auto":
            self.pop_size = 10*self.cf_matrix.shape[1]
        else:
            self.pop_size = int(num_individuals)
        self.change_prob = change_prob
        self.num_genes = self.cf_matrix.shape[1]
        self.individuals = self._initialize_individuals(max_num_in_init_pool)
        self.fitness = np.zeros(len(self.individuals))
        self.regression = Tikhonov(alpha=alpha, penalize_bias_term=True)
        self.elitism = elitism
        self.mutation_prob = mutation_prob
        self.parallel = parallel
        self.num_cores = num_cores
        self.statistics = {
            "best_cv": [],
            "worst_cv": []
        }
        self.evaluate_fitness()
        self.local_decline = local_decline

    def _initialize_individuals(self, max_num):
        """Initialize a random population."""
        from random import shuffle
        individuals = []
        if os.path.exists(self.fname):
            individ_from_file = np.loadtxt(self.fname,
                                           delimiter=",").astype(int)
            for i in range(individ_from_file.shape[0]):
                individuals.append(individ_from_file[i, :])
        else:
            max_num = max_num or self.num_genes
            indices = list(range(self.num_genes))
            for _ in range(self.pop_size):
                shuffle(indices)
                individual = np.zeros(self.num_genes, dtype=np.uint8)
                num_non_zero = np.random.randint(low=3, high=max_num)
                indx = indices[:num_non_zero]
                individual[np.array(indx)] = 1
                individuals.append(individual)
        return individuals

    def fit_individual(self, individual):
        X = self.cf_matrix[:, individual == 1]
        y = self.e_dft
        coeff = self.regression.fit(X, y)

        e_pred = X.dot(coeff)
        delta_e = y - e_pred

        # precision matrix
        prec = self.regression.precision_matrix(X)
        cv_sq = np.mean((delta_e / (1 - np.diag(X.dot(prec).dot(X.T))))**2)
        return coeff, 1000.0*np.sqrt(cv_sq)

    def evaluate_fitness(self):
        """Evaluate fitness of all species."""

        if self.parallel:
            num_cores = self.num_cores or int(mp.cpu_count()/2)
            args = [(self, indx) for indx in range(len(self.individuals))]
            workers = mp.Pool(num_cores)
            self.fitness[:] = workers.map(eval_fitness, args)
        else:
            for i, ind in enumerate(self.individuals):
                _, cv = self.fit_individual(ind)
                self.fitness[i] = 1.0/cv

    def flip_mutation(self, individual):
        """Apply mutation operation."""
        rand_num = np.random.rand(len(individual))
        flip_indx = (rand_num < self.change_prob)
        individual[flip_indx] = (individual[flip_indx]+1) % 2
        return individual

    def sparsify_mutation(self, individual):
        """Change one 1 to 0."""
        indx = np.argwhere(individual == 1)
        rand_num = np.random.rand(len(indx))
        flip_indx = (rand_num < self.change_prob)
        individual[indx[flip_indx]] = 0
        return individual

    def make_valid(self, individual):
        """Make sure that there is at least two active ECIs."""
        if np.sum(individual) < 2:
            while np.sum(individual) < 2:
                indx = np.random.randint(low=0, high=len(individual))
                individual[indx] = 1
        return individual

    def create_new_generation(self):
        """Create a new generation."""
        from random import choice
        new_generation = []
        srt_indx = np.argsort(self.fitness)[::-1]

        assert self.fitness[srt_indx[0]] >= self.fitness[srt_indx[1]]
        mutation_type = ["flip", "sparsify"]

        # Pass the fittest to the next generation
        for i in range(self.elitism):
            individual = self.individuals[srt_indx[i]].copy()
            new_generation.append(individual)

            # Try to insert mutated versions of the best
            # solutions
            mut_type = choice(mutation_type)
            if mut_type == "flip":
                individual = self.flip_mutation(individual.copy())
            else:
                individual = self.sparsify_mutation(individual.copy())
            new_generation.append(self.make_valid(individual))

        cumulative_sum = np.cumsum(self.fitness)
        cumulative_sum /= cumulative_sum[-1]
        num_inserted = len(new_generation)

        # Create new generation by mergin existing
        for i in range(num_inserted, self.pop_size):
            rand_num = np.random.rand()
            p1 = np.argmax(cumulative_sum > rand_num)
            p2 = p1
            while p2 == p1:
                rand_num = np.random.rand()
                p2 = np.argmax(cumulative_sum > rand_num)

            crossing_point = np.random.randint(low=0, high=self.num_genes)
            new_individual = self.individuals[p1].copy()
            new_individual[crossing_point:] = \
                self.individuals[p2][crossing_point:]

            new_individual2 = self.individuals[p2].copy()
            new_individual2[crossing_point:] = \
                self.individuals[p1][crossing_point:]
            if np.random.rand() < self.mutation_prob:
                mut_type = choice(mutation_type)
                if mut_type == "flip":
                    new_individual = self.flip_mutation(new_individual)
                    new_individual2 = self.flip_mutation(new_individual2)
                else:
                    new_individual = self.sparsify_mutation(new_individual)
                    new_individual2 = self.sparsify_mutation(new_individual2)

            if len(new_generation) <= len(self.individuals)-2:
                new_generation.append(self.make_valid(new_individual))
                new_generation.append(self.make_valid(new_individual2))
            elif len(new_generation) == len(self.individuals)-1:
                new_generation.append(self.make_valid(new_individual))
            else:
                break
        self.individuals = new_generation

    def population_diversity(self):
        """Check the diversity of the population."""
        std = np.std(self.individuals)
        return np.mean(std)

    def log(self, msg, end="\n"):
        """Log messages."""
        print(msg, end=end)

    @property
    def best_individual(self):
        best_indx = np.argmax(self.fitness)
        individual = self.individuals[best_indx]
        return individual

    @property
    def best_cv(self):
        return 1.0/np.max(self.fitness)

    @property
    def best_individual_indx(self):
        best_indx = np.argmax(self.fitness)
        return best_indx

    @staticmethod
    def get_instance_array():
        raise TypeError("Does not make sense to create an instance array GA.")

    @property
    def selected_cluster_names(self):
        from itertools import compress
        individual = self.best_individual
        return list(compress(self.cluster_names, individual))

    def save_population(self):
        # Save population
        np.savetxt(self.fname, self.individuals, delimiter=",", fmt="%d")
        print("\nPopulation written to {}".format(self.fname))

    def save_cluster_names(self):
        """Store cluster names of best population to file."""
        with open(self.fname_cluster_names, 'w') as out:
            for name in self.selected_cluster_names:
                out.write(name+"\n")
        print("Selected cluster names saved to {}".format(self.fname_cluster_names))

    def plot_evolution(self):
        """Create a plot of the evolution."""
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.statistics["best_cv"], label="best")
        ax.plot(self.statistics["worst_cv"], label="worst")
        ax.set_xlabel("Generation")
        ax.set_ylabel("CV score (meV/atom)")
        plt.show()

    def run(self, gen_without_change=1000, min_change=0.01, save_interval=100):
        """Run the genetic algorithm.

        Arguments
        ===========
        gen_without_change: int
            Terminate if gen_without_change are created without sufficient
            improvement
        min_change: float
            Changes a larger than this value is considered "sufficient"
            improvement
        save_interval: int
            Rate at which all the populations are backed up in a file
        """
        num_gen_without_change = 0
        current_best = 0.0
        gen = 0
        while(True):
            self.evaluate_fitness()

            best_indx = np.argmax(self.fitness)

            # Start to perform local optimization on the best individual
            # after the earliest possible return
            # If local optimization is turned on too early it seems like
            # it is easy to reach premature convergence
            if best_indx != 0 and self.local_decline and \
                    gen >= gen_without_change:
                self.log("Performing local optimization on new "
                         "best candidate.")
                self._local_optimization()
            cv = 1.0/self.fitness[best_indx]
            num_eci = np.sum(self.individuals[best_indx])
            diversity = self.population_diversity()
            self.statistics["best_cv"].append(1.0/np.max(self.fitness))
            self.statistics["worst_cv"].append(1.0/np.min(self.fitness))
            self.log("Generation: {}. Best CV: {:.2f} meV/atom "
                     "Num ECI: {}. Pop. div: {:.2f}"
                     "".format(gen, cv, num_eci, diversity), end="\r")
            self.create_new_generation()

            if abs(current_best - cv) > min_change:
                num_gen_without_change = 0
            else:
                num_gen_without_change += 1
            current_best = cv

            if gen % save_interval == 0:
                self.save_population()
                self.save_cluster_names()

            if num_gen_without_change >= gen_without_change:
                self.log("\nReached {} generations without sufficient "
                         "improvement".format(gen_without_change))
                break
            gen += 1

        if self.local_decline:
            # Perform a last local optimization
            self._local_optimization()
        self.save_population()
        self.save_cluster_names()
        return self.selected_cluster_names

    def _local_optimization(self, indx=None):
        """Perform a local optimization strategy to the best individual."""
        from random import choice
        from copy import deepcopy
        if indx is None:
            individual = self.best_individual
        else:
            individual = self.individuals[indx]

        num_steps = 10*len(individual)
        cv_min = self.best_cv
        for _ in range(num_steps):
            flip_indx = choice(range(len(individual)))
            individual_cpy = deepcopy(individual)
            individual_cpy[flip_indx] = (individual_cpy[flip_indx]+1) % 2
            _, cv = self.fit_individual(individual_cpy)

            if cv < cv_min:
                cv_min = cv
                individual = individual_cpy

        for i in range(len(self.individuals)):
            if np.allclose(individual, self.individuals[i]):
                # The individual already exists in the population so we don't
                # insert it
                return

        self.individuals[self.best_individual_indx] = individual


def eval_fitness(args):
    ga = args[0]
    indx = args[1]
    _, cv = ga.fit_individual(ga.individuals[indx])
    return 1.0/cv
