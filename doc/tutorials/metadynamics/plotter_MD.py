# creates: MD.png

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys


p = subprocess.Popen(sys.executable + ' MD.py',
                     shell=True, stdout=subprocess.PIPE)
p.wait()

cvs = np.loadtxt('COLVAR', usecols=[1, 2], unpack=True)

fig, ax = plt.subplots(figsize=(7, 8))

ax.plot(cvs[0], cvs[1], ".", markersize=5, label='MD-trajectory')

isomers = [[0.747186,
            0.957874,
            0.591272,
            0.755238],
           [1.318370,
            0.298621,
            -0.11596,
            0.350568]]

ax.tick_params(axis='y', labelsize=25)
ax.tick_params(axis='x', labelsize=25)

ax.plot(isomers[0], isomers[1], 'o', markersize=10, color='white')
ax.plot(isomers[0], isomers[1], '*', markersize=6, color='red',
        label='Isomers')

ax.set_xlabel('SCM', fontsize=40)
ax.set_ylabel('TCM', fontsize=40, labelpad=-20)
ax.set_xlim([0.3, 1.2])
ax.set_ylim([-0.35, 1.56])
plt.tight_layout()
plt.legend(fontsize=20, loc=4)
fig.savefig('MD.png')

subprocess.Popen('rm -f bck.*', shell=True, stdout=subprocess.PIPE)
