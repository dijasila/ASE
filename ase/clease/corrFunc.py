"""Module for calculating correlation functions."""
import numpy as np
from ase.atoms import Atoms
from ase.clease import CEBulk, CECrystal
from ase.clease.tools import wrap_and_sort_by_position, equivalent_deco
from ase.db import connect
import multiprocessing as mp
#from numba import jit
from ase.clease.jit import jit

# workers can not be a member of CorrFunction since CorrFunctions is passed
# as argument to the map function. Hence, we leave it as a global variable,
# but it is initialized wheh the CorrFunction object is initialized.
workers = None


class CorrFunction(object):
    """Calculate the correlation function.

    Arguments
    =========
    setting: settings object

    parallel: bool (optional)
        specify whether or not to use the parallel processing for ``get_cf''
        method.

    num_core: int or "all" (optional)
        specify the number of cores to use for parallelization.
    """

    def __init__(self, setting, parallel=False, num_core="all"):
        if not isinstance(setting, (CEBulk, CECrystal)):
            raise TypeError("setting must be CEBulk or CECrystal "
                            "object")
        self.setting = setting

        self.parallel = parallel
        self.num_core = num_core

        bf = self.setting.basis_functions
        self.symb_id = {}
        for i, symb in enumerate(bf[0].keys()):
            self.symb_id[symb] = i

        self.bf_npy = np.zeros((len(bf), len(self.symb_id.keys())))
        self.tm = self._full_trans_matrix()
        for i, item in enumerate(bf):
            for k, v in self.symb_id.items():
                self.bf_npy[i, v] = item[k]

        if parallel:
            global workers
            if self.num_core == "all":
                num_proc = int(mp.cpu_count()/2)
            else:
                num_proc = int(self.num_core)
            if workers is None:
                workers = mp.Pool(num_proc)

    def _atoms2npy(self, atoms):
        array = np.zeros(len(atoms), dtype=np.int32)
        for atom in atoms:
            array[atom.index] = self.symb_id.get(atom.symbol, -1)
        return array

    def _full_trans_matrix(self):
        num_atoms = len(self.setting.trans_matrix)
        tm = np.zeros((num_atoms, num_atoms), dtype=np.int32)
        for i, row in enumerate(self.setting.trans_matrix):
            for k, v in row.items():
                tm[i, k] = v
        return tm

    def get_c1(self, atoms, dec):
        """Get correlation function for single-body clusters."""
        c1 = 0
        for element, spin in self.setting.basis_functions[dec].items():
            num_element = len([a for a in atoms if a.symbol == element])
            c1 += num_element * spin
        c1 /= float(len(atoms))
        return c1

    def get_cf(self, atoms, return_type='dict'):
        """Calculate correlation function for all possible clusters.

        Arguments:
        =========
        atoms: Atoms object

        return_type: str
            -'dict' (default): returns a dictionary (e.g., {'name': cf_value})
            -'tuple': returns a list of tuples (e.g., [('name', cf_value)])
            -'array': NumPy array containing *only* the correlation function
                      values in the same order as the order in
                      "setting.full_cluster_names"
        """
        if not isinstance(atoms, Atoms):
            raise TypeError('atoms must be an Atoms object')

        bf_list = list(range(len(self.setting.basis_functions)))
        cf = {}
        # ----------------------------------------------------
        # Compute correlation function up the max_cluster_size
        # ----------------------------------------------------
        # loop though all cluster sizes
        cf['c0'] = 1.0

        # Update singlets
        for dec in bf_list:
            cf['c1_{}'.format(dec)] = self.get_c1(atoms, dec)

        cnames = self.setting.cluster_names
        if self.parallel:
            args = [(self, atoms, name) for name in cnames]
            res = workers.map(get_cf_parallel, args)
            cf = {}
            for r in res:
                cf[r[0][0]] = r[0][1]
            if return_type == 'tuple':
                return list(cf.items())
            elif return_type == 'array':
                cf = np.array([cf[x] for x in cnames], dtype=float)
            return cf
        return self.get_cf_by_cluster_names(atoms, cnames, return_type)

    def get_cf_by_cluster_names(self, atoms, cluster_names,
                                return_type='dict'):
        """Calculate correlation functions of the specified clusters.

        Arguments
        =========
        atoms: Atoms object

        cluster_names: list
            names (str) of the clusters for which the correlation functions are
            calculated for the structure provided in atoms

        return_type: str
            -'dict' (default): returns a dictionary (e.g., {'name': cf_value})
            -'tuple': returns a list of tuples (e.g., [('name', cf_value)])
            -'array': NumPy array containing *only* the correlation function
                      values in the same order as the order provided in the
                      "cluster_names"
        """
        if isinstance(atoms, Atoms):
            self.check_cell_size(atoms)
        else:
            raise TypeError('atoms must be Atoms object')
        cf = {}

        atoms_npy = self._atoms2npy(atoms)
        self.tm = self._full_trans_matrix()
        for name in cluster_names:
            if name == 'c0':
                cf[name] = 1.
                continue
            prefix = name.rpartition('_')[0]
            dec = name.rpartition('_')[-1]
            dec_list = [int(i) for i in dec]
            # find c{num} in cluster type
            n = int(prefix[1])

            if n == 1:
                cf[name] = self.get_c1(atoms, int(dec))
                continue

            sp = 0.
            count = 0
            # loop through the symmetry inequivalent groups
            # for symm in range(self.num_trans_symm):
            for cluster_set in self.setting.cluster_info:
                if prefix not in cluster_set.keys():
                    continue
                cluster = cluster_set[prefix]
                sp_temp, count_temp = \
                    self._spin_product(atoms_npy, cluster, dec_list)
                sp += sp_temp
                count += count_temp
            cf_temp = sp / count
            cf['{}_{}'.format(prefix, dec)] = cf_temp

        if return_type == 'dict':
            pass
        elif return_type == 'tuple':
            cf = list(cf.items())
        elif return_type == 'array':
            cf = np.array([cf[x] for x in cluster_names], dtype=float)
        return cf

    def reconfigure_db_entries(self, select_cond=None):
        """Reconfigure the correlation function values of the entries in DB.

        Arguments
        =========
        select_cond: list
            -None (default): select every item in DB with
                             "struct_type='initial'"
            -else: select based on the condictions provided
                  (struct_type='initial' is not automatically included)
        """
        db = connect(self.setting.db_name)
        select = []
        if select_cond is not None:
            for cond in select_cond:
                select.append(cond)
        else:
            select = [('struct_type', '=', 'initial')]

        # get how many entries need to be reconfigured
        row_ids = [row.id for row in db.select(select)]
        num_reconf = len(row_ids)
        print('{} entries will be reconfigured'.format(num_reconf))

        count = 0
        for row_id in row_ids:
            row = db.get(id=row_id)
            kvp = row.key_value_pairs

            # delete existing CF values
            keys = []
            for key in kvp.keys():
                if key.startswith(('c0', 'c1', 'c2', 'c3', 'c4', 'c5',
                                   'c6', 'c7', 'c8', 'c9')):
                    keys.append(key)
            db.update(row_id, delete_keys=keys)

            # get new CF based on setting
            atoms = wrap_and_sort_by_position(row.toatoms())
            cf = self.get_cf(atoms, return_type='dict')
            db.update(row_id, **cf)
            count += 1
            print('updated {} of {} entries'.format(count, num_reconf))

    def _spin_product(self, atoms, cluster, deco):
        """Get spin product of a given cluster.

        Arguments:
        =========
        atoms: Atoms object

        cluster: dict
            A dictionary containing all necessary information about the
            family of cluster (i.e., list of indices, order, equivalent sites,
            symmetry group, etc.).

        deco: tuple
            Decoration number that specifies which basis function should be
            used for getting the spin variable of each atom.
        """
        sp = 0.
        count = 0

        # spin product of each atom in the symmetry equivalent group
        indices_of_symm_group = \
            self.setting.index_by_trans_symm[cluster["symm_group"]]
        ref_indx_grp = indices_of_symm_group[0]

        eq_sites = list(cluster["equiv_sites"])
        equiv_deco = np.array(equivalent_deco(deco, eq_sites))

        # Convert to numpy arrays
        indices = np.array(cluster["indices"])
        order = np.array(cluster["order"])

        for ref_indx in indices_of_symm_group:
            sp_temp, count_temp = _sp_same_shape_deco_for_ref_indx_jit(
                atoms, ref_indx, indices, order, ref_indx_grp, equiv_deco,
                self.tm, self.bf_npy)
            sp += sp_temp
            count += count_temp
        return sp, count

    def _spin_product_one_cluster(self, atoms, ref_indx, cluster_indices,
                                  order, ref_indx_grp, equiv_deco, bf, tm):
        """Compute spin product for one cluster (same shape, deco, ref_indx).

        Arguments
        =========
        atoms: np.ndarray
            1D numpy array representation of the atoms object. Each symbol
            has a unique number set by the CorrFunction class. Example: If
            curr_func.symb_id = {"Al": 0, "Cu": 1, "Li": 2} and the symbols
            are ["Al", "Al", "Li", "Cu", "Li"], this array would be
            [0, 0, 2, 1, 2]

        ref_indx: int
            Index of the atom used as a reference to get clusters.

        cluster_indices: list
            A list where indices of the atoms that consistute a cluster.

        order: list
            A list of how the indices in "cluster_indices" should be ordered.
            The indices of atoms are sorted in a decrease order of internal
            distances to other members of the cluster.

        eq_sites: list
            A list that groups the equivalent atoms in a cluster. Atoms are
            classified as equivalent when they are inditinguishable based on
            the geometry of the cluster.
            (e.g., equilateral triangles have 3 indistinguishable points.)

        ref_indx_grp: int
            Index of the reference atom used for the translational symmetry
            group.

        deco: tuple
            Decoration number that specifies which basis function should be
            used for getting the spin variable of each atom.
        """
        return _spin_product_one_cluster_jit(atoms, ref_indx, cluster_indices,
                                             order, ref_indx_grp, equiv_deco,
                                             tm, bf)

    def check_cell_size(self, atoms):
        """Check the size of provided cell and create a template if necessary.

        Arguments:
        =========
        atoms: Atoms object
            *Unrelaxed* structure
        """
        self.setting.set_active_template(atoms=atoms, generate_template=True)
        return atoms


def get_cf_parallel(args):
    cf = args[0]
    atoms = args[1]
    name = args[2]
    return cf.get_cf_by_cluster_names(atoms, [name], return_type="tuple")


@jit(nopython=True)
def _spin_product_one_cluster_jit(atoms, ref_indx, cluster_indices,
                                  order, ref_indx_grp, equiv_deco,
                                  trans_matrix, bf):
        """Compute sp of cluster with same shape and deco for given ref atom.

        Arguments
        =========
        atoms: np.ndarray
            1D numpy array representation of the atoms object. Each symbol
            has a unique number set by the CorrFunction class. Example: If
            curr_func.symb_id = {"Al": 0, "Cu": 1, "Li": 2} and the symbols
            are ["Al", "Al", "Li", "Cu", "Li"], this array would be
            [0, 0, 2, 1, 2]

        ref_indx: int
            Index of the atom used as a reference to get clusters.

        cluster_indices: np.ndarray
            1D numpy array of inidices that constitue a cluster. Each row
            correspond to a sub-cluster. For a triplet this could be
            [0, 1]
            hence the reference index itself is not included.

        indx_order: np.ndarray
            A 1D array of how the indices in "cluster_indices" should be
            ordered. The indices of atoms are sorted in a decrease order of
            internal distances to other members of the cluster.

        ref_indx_grp: int
            Index of the reference atom used for the translational symmetry
            group.

        deco: np.ndarray
            Decoration number that specifies which basis function should be
            used for getting the spin variable of each atom. Each row in the
            array represents the different combination that by symmetry should
            be averaged.
        trans_matrix: np.ndarray
            2D Numpy array of the full translation matrix.
        bf: np.ndarray
            2D Numpy array of holding the basis functions.
        """
        count = 0
        sp = 0.0
        indices = np.zeros(len(cluster_indices) + 1)
        indices[0] = ref_indx_grp
        for i in range(len(cluster_indices)):
            indices[i+1] = cluster_indices[i]

        sorted_indices = np.zeros(len(indices), dtype=np.int32)
        for i in range(len(indices)):
            sorted_indices[i] = indices[order[i]]

        for dec_num in range(equiv_deco.shape[0]):
            dec = equiv_deco[dec_num, :]
            sp_temp = 1.0
            # loop through indices of atoms in each cluster
            for i in range(len(sorted_indices)):
                indx = sorted_indices[i]
                trans_indx = trans_matrix[ref_indx, indx]
                sp_temp *= bf[dec[i], atoms[trans_indx]]
            sp += sp_temp
            count += 1
        num_equiv = float(equiv_deco.shape[0])
        return sp/num_equiv, count/num_equiv


@jit(nopython=True)
def _sp_same_shape_deco_for_ref_indx_jit(atoms, ref_indx, indices, order,
                                         ref_indx_grp, equiv_deco, tm, bf):
        """Compute sp of cluster with same shape and deco for given ref atom.

        Arguments
        =========
        atoms: np.ndarray
            1D numpy array representation of the atoms object. Each symbol
            has a unique number set by the CorrFunction class. Example: If
            curr_func.symb_id = {"Al": 0, "Cu": 1, "Li": 2} and the symbols
            are ["Al", "Al", "Li", "Cu", "Li"], this array would be
            [0, 0, 2, 1, 2]

        ref_indx: int
            Index of the atom used as a reference to get clusters.

        indx_list: np.ndarray
            2D numpy array of inidices that constitue a cluster. Each row
            correspond to a sub-cluster. For a triplet this could be
            [[0, 1]
             [5, 2]
             [7, 8]]
             hence the reference index itself is not included.

        indx_order: np.ndarray
            A 2D array of how the indices in "indx_list" should be ordered.
            The indices of atoms are sorted in a decrease order of internal
            distances to other members of the cluster.

        ref_indx_grp: int
            Index of the reference atom used for the translational symmetry
            group.

        deco: np.ndarray
            Decoration number that specifies which basis function should be
            used for getting the spin variable of each atom. Each row in the
            array represents the different combination that by symmetry should
            be averaged.
        tm: np.ndarray
            2D Numpy array of the full translation matrix.
        bf: np.ndarray
            2D Numpy array of holding the basis functions.
        """
        count = 0
        sp = 0.0
        for i in range(indices.shape[0]):
            temp_sp, temp_cnt = \
                _spin_product_one_cluster_jit(atoms, ref_indx,
                                              indices[i, :], order[i, :],
                                              ref_indx_grp, equiv_deco, tm, bf)
            sp += temp_sp
            count += temp_cnt
        return sp, count
