float_keys = [
    "charge",
    "charge_mix_param",
    "default_initial_moment",
    "fixed_spin_moment",
    "hartree_convergence_parameter",
    "harmonic_length_scale",
    "ini_linear_mix_param",
    "ini_spin_mix_parma",
    "initial_moment",
    "MD_MB_init",
    "MD_time_step",
    "prec_mix_param",
    "set_vacuum_level",
    "spin_mix_param",
]

exp_keys = [
    "basis_threshold",
    "occupation_thr",
    "sc_accuracy_eev",
    "sc_accuracy_etot",
    "sc_accuracy_forces",
    "sc_accuracy_rho",
    "sc_accuracy_stress",
]

string_keys = [
    "communication_type",
    "density_update_method",
    "KS_method",
    "mixer",
    "output_level",
    "packed_matrix_format",
    "relax_unit_cell",
    "restart",
    "restart_read_only",
    "restart_write_only",
    "spin",
    "total_energy_method",
    "qpe_calc",
    "xc",
    "species_dir",
    "run_command",
    "plus_u",
]

int_keys = [
    "empty_states",
    "ini_linear_mixing",
    "max_relaxation_steps",
    "max_zeroin",
    "multiplicity",
    "n_max_pulay",
    "sc_iter_limit",
    "walltime",
]

bool_keys = [
    "collect_eigenvectors",
    "compute_forces",
    "compute_kinetic",
    "compute_numerical_stress",
    "compute_analytical_stress",
    "compute_heat_flux",
    "distributed_spline_storage",
    "evaluate_work_function",
    "final_forces_cleaned",
    "hessian_to_restart_geometry",
    "load_balancing",
    "MD_clean_rotations",
    "MD_restart",
    "override_illconditioning",
    "override_relativity",
    "restart_relaxations",
    "squeeze_memory",
    "symmetry_reduced_k_grid",
    "use_density_matrix",
    "use_dipole_correction",
    "use_local_index",
    "use_logsbt",
    "vdw_correction_hirshfeld",
]

list_keys = [
    "init_hess",
    "k_grid",
    "k_offset",
    "MD_run",
    "MD_schedule",
    "MD_segment",
    "mixer_threshold",
    "occupation_type",
    "output",
    "cube",
    "preconditioner",
    "relativistic",
    "relax_geometry",
]
