# creates: fes.png, MTD.png

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys


p = subprocess.Popen(sys.executable + ' MTD.py',
                     shell=True, stdout=subprocess.PIPE)
p.wait()

p = subprocess.Popen("plumed sum_hills --hills HILLS --outfile fes.dat" +
                     " --bin 300,300 --min 0.3,-0.35 --max 1.2,1.56",
                     shell=True, stdout=subprocess.PIPE)
p.wait()

isomers = [[0.747186,
            0.957874,
            0.591272,
            0.755238],
           [1.318370,
            0.298621,
            -0.11596,
            0.350568]]

# Figure 1

scm = np.loadtxt('fes.dat', usecols=0).reshape(301, 301)
tcm = np.loadtxt('fes.dat', usecols=1).reshape(301, 301)
fes = np.loadtxt('fes.dat', usecols=2).reshape(301, 301)
minimum = min(fes.flatten())

fig, ax = plt.subplots(figsize=(9, 9))

ax.tick_params(axis='y', labelsize=25)
ax.tick_params(axis='x', labelsize=25)

ax.plot(isomers[0], isomers[1], 'o', markersize=8, color='white')
ax.plot(isomers[0], isomers[1], '*', markersize=4.8, color='red')

im = ax.contourf(scm, tcm, fes, np.arange(minimum, 0.1, 0.1), cmap='magma',
                 alpha=0.85)
cp = ax.contour(scm, tcm, fes, np.arange(minimum, 0.1, 0.1), linestyles='-',
                colors='darkgray', linewidths=1.2)

cbar = fig.colorbar(im, ax=ax, format='%1.1f', pad=0.03, shrink=0.9)
cbar.set_label(label=r'FES[$\epsilon$]', fontsize=40)
cbar.set_ticks(np.arange(minimum, 0.1, 0.2))
cbar.ax.tick_params(labelsize=25)

ax.set_xlabel('SCM', fontsize=40)
ax.set_ylabel('TCM', fontsize=40, labelpad=-15)
plt.tight_layout()
plt.savefig('fes.png')

# Figure 2

cvs = np.loadtxt('HILLS', usecols=[1, 2], unpack=True)

fig, ax = plt.subplots(figsize=(7, 8))

ax.plot(cvs[0], cvs[1], ".", markersize=5, label='MTD-trajectory')

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
fig.savefig('MTD.png')

subprocess.Popen('rm -f bck.*', shell=True, stdout=subprocess.PIPE)
