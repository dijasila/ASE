from matplotlib.testing.compare import compare_images
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from ase.lattice import HEX2D


datadir = Path(__file__).parent


def test_repeat_transpose_bz():
    """Testint plot_bz

    As I could not write the unit test by comparing the images, I take the another way to test.
    """

    hex2d = HEX2D(a=1.0)
    r = Rotation.from_rotvec([0, 0, 10], degrees=True)
    fig, ax = plt.subplots()
    hex2d.plot_bz(repeat=(2, 1), transforms=[r])
    fig.savefig(datadir / 'test_bz.png')
    img1 = datadir / 'baseline' / 'rotated_bz.png'
    img2 = datadir / 'test_bz.png'
    compare_images(str(img1), str(img2), 0.1)
