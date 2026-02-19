'''
Python script to create plots from the solution of a
PIAFS simulation.

- if op_overwrite is set to "no", a plot is generated
for each reacting species and each
simulation time for which the solution is available.
- if op_overwrite is set to "yes", a single plot is
created for for each reacting species.
- solution must be in binary format

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

figsize=(15,9)
plt_dir_name='plots'

'''
Set up the simulation parameters
'''
sim_path = '.'
sim_inp_data = piafsutils.readPIAFSInpFile(sim_path+'/simulation.inp')
solver_inp_data = piafsutils.readPIAFSInpFile(sim_path+'/solver.inp')
chem_inp_data = piafsutils.readPIAFSInpFile(sim_path+'/chemistry.inp')

if not sim_inp_data:
    nsims = 1
else:
    nsims = int(sim_inp_data['nsims'][0])

z_mm = 0.0
z_i = 0
nz = 1
if chem_inp_data:
  z_mm = float(chem_inp_data['z_mm'])
  L_z = float(chem_inp_data['Lz'])
  nz = int(chem_inp_data['nz'])
  z_i = int(z_mm*0.001*nz/L_z)
nz = z_i + 1


print('Number of z-layers: ', nz)

if solver_inp_data['op_file_format'] != 'binary':
    raise Exception("op_file_format must be 'binary' in solver.inp.")

ndims = int(solver_inp_data['ndims'][0])
size = np.int32(solver_inp_data['size'])

nspecies = 8

if not os.path.exists(plt_dir_name):
      os.makedirs(plt_dir_name)

for zi in range(nz):

  if nz >= 100:
    op_prefix = 'op_species_z' + f'{zi:03d}'
  elif nz >= 10:
    op_prefix = 'op_species_z' + f'{zi:02d}'
  elif nz >= 1:
    op_prefix = 'op_species_z' + f'{zi:01d}'

  if solver_inp_data['op_overwrite'] == 'no':

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
    print('  nspecies = ', nspecies)
    print('  grid size = ', size)
    print('  simulation dt = ', dt)
    print('  niter = ', niter)
    print('  final time = ', t_final)
    print('  snapshot dt = ', dt_snapshots)
    print('  expected number of snapshots = ', n_snapshots)

    '''
    Load simulation data (solution snapshots)
    '''
    x,solution_snapshots = piafsutils.getSolutionSnapshots( sim_path,
                                                              nsims,
                                                              n_snapshots,
                                                              ndims,
                                                              nspecies,
                                                              size,
                                                              op_prefix )
    solution_snapshots = np.float32(solution_snapshots)
    n_snapshots = solution_snapshots.shape[0]
    print('  number of snapshots = ', n_snapshots)

    for var in range(nspecies):
      for i in range(n_snapshots):
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.set( xlim=(np.min(x), np.max(x)),
                ylim=(np.min(solution_snapshots[:,var::nspecies]),
                      np.max(solution_snapshots[:,var::nspecies]) ) )
        for s in range(nsims):
            solution_snapshots_sim = solution_snapshots[s*n_snapshots:(s+1)*n_snapshots]
            ax.plot(x, solution_snapshots_sim[i, var::nspecies], lw=2)
        ax.set_title('species {:}, t={:.3}'.format(var,i*dt_snapshots))
        plt.grid(visible=True, linestyle=':', linewidth=1)
        plt_fname = plt_dir_name+'/fig_'+f'z{zi:02d}'+'_'+f'{var:02d}'+'_'+f'{i:05d}'+'.png'
        print('Saving %s' % plt_fname)
        plt.savefig(plt_fname)
        plt.close()

  else:

    niter = int(solver_inp_data['n_iter'][0])
    dt = float(solver_inp_data['dt'][0])
    t_final = dt*niter

    n_snapshots = 1

    print('Simulation parameters:')
    print('  ndims = ', ndims)
    print('  nspecies = ', nspecies)
    print('  grid size = ', size)
    print('  final time = ', t_final)
    print('  number of snapshots = ', n_snapshots)

    '''
    Load simulation data (solution snapshots)
    '''
    x,solution_snapshots = piafsutils.getSolutionSnapshots( sim_path,
                                                              nsims,
                                                              n_snapshots,
                                                              ndims,
                                                              nspecies,
                                                              size,
                                                              op_prefix )
    solution_snapshots = np.float32(solution_snapshots)

    for var in range(nspecies):
      fig = plt.figure(figsize=figsize)
      ax = plt.axes()
      ax.set( xlim=(np.min(x), np.max(x)),
              ylim=(np.min(solution_snapshots[:,var::nspecies]),
                    np.max(solution_snapshots[:,var::nspecies]) ) )
      for s in range(nsims):
          solution_snapshots_sim = solution_snapshots[s*n_snapshots:(s+1)*n_snapshots]
          ax.plot(x, solution_snapshots_sim[0, var::nspecies], lw=2)
      ax.set_title('species {:}, t={:.3}'.format(var,t_final))
      plt.grid(visible=True, linestyle=':', linewidth=1)
      plt_fname = plt_dir_name+'/fig_'+f'z{zi:02d}'+'_'+f'{var:02d}'+'.png'
      print('Saving %s' % plt_fname)
      plt.savefig(plt_fname)
      plt.close()

