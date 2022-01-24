# creates: trajectory.png

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys


subprocess.check_output(sys.executable + ' MD.py', shell=True)
subprocess.check_output(sys.executable + ' MTD.py', shell=True)
subprocess.check_output(sys.executable + ' postpro.py', shell=True)

cvs_md = np.loadtxt('COLVAR', usecols=[1, 2], unpack=True)
cvs_mtd = np.loadtxt('HILLS', usecols=[1, 2], unpack=True)

fig, ax = plt.subplots(figsize=(6, 8))

ax.plot(cvs_md[0], cvs_md[1], ".", markersize=5, label='MD-trajectory')
ax.plot(cvs_mtd[0], cvs_mtd[1], ".", markersize=5, label='MTD-trajectory')

isomers = [[0.747186,
            0.957874,
            0.591272,
            0.755238],
           [1.318370,
            0.298621,
            -0.11596,
            0.350568]]

ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=15)

ax.plot(isomers[0], isomers[1], 'o', markersize=10, color='white')
ax.plot(isomers[0], isomers[1], '*', markersize=6, color='red',
        label='Isomers')

ax.set_xlabel('SCM', fontsize=20)
ax.set_ylabel('TCM', fontsize=20, labelpad=-10)
ax.set_xlim([0.5, 1.1])
ax.set_ylim([-0.25, 1.5])
plt.tight_layout()
plt.legend(fontsize=15, loc=4)
fig.savefig('trajectory.png')

subprocess.Popen('rm -f bck.*', shell=True, stdout=subprocess.PIPE)
