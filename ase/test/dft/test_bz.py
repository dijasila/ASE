from matplotlib.testing.compare import compare_images
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numpy as np
from ase.lattice import HEX2D


datadir = Path(__file__).parent


def test_repeat_transpose_bz():
    """Testint plot_bz."""

    hex2d = HEX2D(a=1.0)
    r = Rotation.from_rotvec([0, 0, np.deg2rad(10)])
    fig, ax = plt.subplots()
    hex2d.plot_bz(repeat=(2, 1), transforms=[r], ax=ax)
    fig.savefig(datadir / 'test_bz.png')
    img1 = datadir / 'baseline' / 'rotated_bz.png'
    img2 = datadir / 'test_bz.png'
    compare_images(str(img1), str(img2), 0.1)
