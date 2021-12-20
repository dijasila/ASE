import numpy as np
import matplotlib.pyplot as plt
import subprocess


# Reconstruct the free energy from HILLS using the plumed tool "sum_hills" :
p = subprocess.Popen("plumed sum_hills --hills HILLS --outfile fes.dat" +
                     " --bin 300,300 --min 0.3,-0.35 --max 1.2,1.56",
                     shell=True, stdout=subprocess.PIPE)
p.wait()

# Import free energy and reshape with the number of bins defined in the
# reconstruction process.
scm = np.loadtxt('fes.dat', usecols=0).reshape(301, 301)
tcm = np.loadtxt('fes.dat', usecols=1).reshape(301, 301)
fes = np.loadtxt('fes.dat', usecols=2).reshape(301, 301)

# Plot free energy surface
fig, ax = plt.subplots(figsize=(9, 9))

im = ax.contourf(scm, tcm, fes, 10, cmap='magma')
cp = ax.contour(scm, tcm, fes, 10)  # add isopotential lines

cbar = fig.colorbar(im, ax=ax)

cbar.set_label(label=r'FES[$\epsilon$]')
ax.set_xlabel('SCM')
ax.set_ylabel('TCM')

plt.show()
