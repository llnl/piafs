'''
Python script to create plots from the solution of a
PIAFS simulation.

x-z plots are generated for each species

Make sure the environment variable "PIAFS_DIR" is set
and points to the correct location (/path/to/piafs)
'''

import os
import time
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

piafs_dir = os.environ.get('PIAFS_DIR')
piafs_dir_python = piafs_dir + '/Examples/Python'
sys.path.append(piafs_dir_python)

import modPIAFSUtils as piafsutils

font = {'size':22}
matplotlib.rc('font', **font)
colormap='jet'

figsize=(50,20)
plt_dir_name='plots'

'''
Set up the simulation parameters
'''
sim_path = '.'
sim_inp_data = piafsutils.readPIAFSInpFile(sim_path+'/simulation.inp')
solver_inp_data = piafsutils.readPIAFSInpFile(sim_path+'/solver.inp')

if not sim_inp_data:
    nsims = 1
else:
    nsims = int(sim_inp_data['nsims'][0])

print('Number of simulations: ', nsims)

if solver_inp_data['op_file_format'] != 'binary':
    raise Exception("op_file_format must be 'binary' in solver.inp.")

ndims = int(solver_inp_data['ndims'][0])
size = np.int32(solver_inp_data['size'])

if solver_inp_data['op_overwrite'] == 'yes':
  print('This script requires time-dependent solution output.')
  exit()

if not os.path.exists(plt_dir_name):
      os.makedirs(plt_dir_name)

niter = int(solver_inp_data['n_iter'][0])
dt = float(solver_inp_data['dt'][0])
t_final = dt*niter

op_write_iter = int(solver_inp_data['file_op_iter'][0])
dt_snapshots = op_write_iter*dt
if (op_write_iter > niter):
  n_snapshots = 2
else:
  n_snapshots = int(niter/op_write_iter) + 1

print('Simulation parameters:')
print('  ndims = ', ndims)
print('  grid size = ', size)
print('  simulation dt = ', dt)
print('  niter = ', niter)
print('  final time = ', t_final)
print('  snapshot dt = ', dt_snapshots)
print('  expected number of snapshots = ', n_snapshots)

'''
Load simulation data (solution snapshots)
'''
op_prefix = 'op_species'
nspecies = 8
grid, solution_snapshots = piafsutils.getSolutionSnapshots( sim_path,
                                                            nsims,
                                                            n_snapshots,
                                                            ndims,
                                                            nspecies,
                                                            size,
                                                            op_prefix )
solution_snapshots = np.float32(solution_snapshots)
n_snapshots = solution_snapshots.shape[0]
print('  number of snapshots = ', n_snapshots)
x = grid[:size[0]]
y = grid[size[0]:size[0]+size[1]]
z = grid[size[0]+size[1]:]
t = np.linspace(0.0, t_final, n_snapshots)
z2d, x2d = np.meshgrid(z, x)

print('3D Domain:');
print(' x: ', np.min(x), np.max(x))
print(' x.shape: ', x.shape)
print(' y: ', np.min(y), np.max(y))
print(' y.shape: ', y.shape)
print(' z: ', np.min(z), np.max(z))
print(' z.shape: ', z.shape)

j0 = 0

for s in range(nsims):
  solution_snapshots_sim = solution_snapshots[s*n_snapshots:(s+1)*n_snapshots]
  solution = np.transpose(solution_snapshots_sim.reshape(n_snapshots,size[2],size[1],size[0],nspecies))
  
  for n in range(n_snapshots):
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=figsize)
    for var in range(nspecies):
      row = var // 4
      col = var % 4
      plot = ax[row, col].pcolormesh(x2d, z2d, solution[var,:,j0,:,n], cmap=colormap, shading='auto')
      ax[row, col].set_title('species {:} at j={:}, y={:.3} t={:.3}'.format(var,j0,y[j0],t[n]))
      fig.colorbar(plot, ax=ax[row, col])
    if nsims > 1:
      plt_fname = plt_dir_name+'/fig_species_'+f'{s:02d}'+'_xz_'+f'j{j0:02d}'+f't{n:03d}'+'.png'
    else:
      plt_fname = plt_dir_name+'/fig_species_xz_'+f'j{j0:02d}'+f't{n:03d}'+'.png'
    print('Saving %s' % plt_fname)
    plt.savefig(plt_fname)
    plt.close()
