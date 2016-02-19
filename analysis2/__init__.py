"""
analysis package for scattering problems on the lattice
"""
# .in_out imports only preliminary, think about more encapsulated solution
from .in_out import inputnames, read_confs, write_data_ascii, confs_subtr, conf_abs, confs_mult
from .correlator import Correlators
from .ensemble import LatticeEnsemble
from .fit import LatticeFit, FitResult
from .plot import LatticePlot
from .functions import func_const, func_ratio, func_single_corr
from .statistics import draw_weighted, compute_error, sys_error, draw_gauss_distributed 
from .interpol import interp_fk
