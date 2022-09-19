from ase.io import read


def get_structure_list(filelist=[], format=None):
    """Get a list of Atoms objects

    Arguments:

    filelist: list
        A list of files

    format: str 
        Used to specify the file-format. Same argument of ase.io.read.
    """
    return [read(file, index=-1, format=format) for file in filelist]


def spap_analysis(atoms_list, i_mode=4, symprec=0.1, **kwargs):
    """Structure Prototype Analysis Package used here to analyze symmetry and compare similarity of large amount of atomic structures.

    Description:

    The structure prototype analysis package (SPAP) can analyze symmetry and compare similarity of a large number of atomic structures. 
    Typically, SPAP can analyze structures predicted by CALYPSO (www.calypso.cn). We use spglib to analyze symmetry. The coordination 
    characterization function (CCF) is used to measure structural similarity. We developed a unique and advanced clustering method to 
    automatically classify structures into groups. If you use this program and method in your research, please read and cite the publication: 

        Su C, Lv J, Li Q, Wang H, Zhang L, Wang Y, Ma Y. Construction of crystal structure prototype database: methods and applications. 
        J Phys Condens Matter. 2017 Apr 26;29(16):165901. doi: 10.1088/1361-648X/aa63cd 


    Installation:

        1. Use command: `pip install spap`
        2. Download the source code from https://github.com/chuanxun/StructurePrototypeAnalysisPackage, then install with command `python3 setup.py install`
        3. We also include module `ccf` and `spap` in ase.calculators.abacus, but these can't be updated quickly with github 

    Parameters
    ----------
    atoms_list: list
        A list of Atoms objects

    i_mode: int
        Different functionality of SPAP.

    symprec: float
        This precision is used to analyze symmetry of atomic structures.

    **kwargs:
        More parameters can be found in ase.calculators.abacus.spap
    """
    try:
        from spap import run_spap
    except ImportError:
        raise ImportError("If you want to use SPAP to analyze symmetry and compare similarity of atomic structures, Please install it first!")

    kwargs.pop('structure_list')
    run_spap(symprec=symprec, structure_list=atoms_list, i_mode=i_mode, **kwargs)
