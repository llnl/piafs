'''
Python script to create plots from the solution of a
PIAFS2D simulation.

op_overwrite must be set to "no": x-t plots are generated
for each primitive variable

Make sure the environment variable "PIAFS2D_DIR" is set
and points to the correct location (/path/to/piafs2d)
'''

import os
import time
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

piafs2d_dir = os.environ.get('PIAFS2D_DIR')
piafs2d_dir_python = piafs2d_dir + '/Examples/Python'
sys.path.append(piafs2d_dir_python)

import modPIAFS2DUtils as piafs2dutils

font = {'size':22}
matplotlib.rc('font', **font)
colormap='jet'

figsize=(12,10)
plt_dir_name='plots'

'''
Set up the simulation parameters
'''
sim_path = '.'
sim_inp_data = piafs2dutils.readPIAFS2DInpFile(sim_path+'/simulation.inp')
solver_inp_data = piafs2dutils.readPIAFS2DInpFile(sim_path+'/solver.inp')

if not sim_inp_data:
    nsims = 1
else:
    nsims = int(sim_inp_data['nsims'][0])

print('Number of simulations: ', nsims)

if solver_inp_data['op_file_format'] != 'binary':
    raise Exception("op_file_format must be 'binary' in solver.inp.")

ndims = int(solver_inp_data['ndims'][0])
nvars = int(solver_inp_data['nvars'][0])
size = np.int32(solver_inp_data['size'])

if not os.path.exists(plt_dir_name):
      os.makedirs(plt_dir_name)

if solver_inp_data['op_overwrite'] == 'yes':
  print('This script requires time-dependent solution output.')
  exit()

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
print('  nvars = ', nvars)
print('  grid size = ', size)
print('  simulation dt = ', dt)
print('  niter = ', niter)
print('  final time = ', t_final)
print('  snapshot dt = ', dt_snapshots)
print('  expected number of snapshots = ', n_snapshots)

'''
Load simulation data (solution snapshots)
'''
x,solution_snapshots = piafs2dutils.getSolutionSnapshots( sim_path,
                                                          nsims,
                                                          n_snapshots,
                                                          ndims,
                                                          nvars,
                                                          size )

t = np.linspace(0.0, t_final, n_snapshots)
y2d, x2d = np.meshgrid(t, x)

solution_snapshots = np.float32(solution_snapshots)
n_snapshots = solution_snapshots.shape[0]
print('  number of snapshots = ', n_snapshots)

for s in range(nsims):
  solution_snapshots_sim = solution_snapshots[s*n_snapshots:(s+1)*n_snapshots]
  cons_sol = np.transpose(solution_snapshots_sim.reshape(n_snapshots,size[0],nvars))[:,:,:]
  rho = cons_sol[0,:,:]
  u = cons_sol[1,:,:] / cons_sol[0,:,:]
  pressure = 0.4 * (cons_sol[2,:,:] - 0.5*cons_sol[1,:,:]*cons_sol[1,:,:]/cons_sol[0,:,:])
  temperature = pressure / rho

  fig = plt.figure(figsize=figsize)
  ax = plt.axes()
  ax.set( xlim=(np.min(x), np.max(x)), ylim=(0, t_final) )
  plot = ax.pcolor(x2d, y2d, rho[:,:], cmap=colormap)
  ax.set_title('rho')
  fig.colorbar(plot, ax=ax)
  if nsims > 1:
    plt_fname = plt_dir_name+'/fig_'+f'{s:02d}'+'_rho.png'
  else:
    plt_fname = plt_dir_name+'/fig_rho.png'
  print('Saving %s' % plt_fname)
  plt.savefig(plt_fname)
  plt.close()

  fig = plt.figure(figsize=figsize)
  ax = plt.axes()
  ax.set( xlim=(np.min(x), np.max(x)), ylim=(0, t_final) )
  plot = ax.pcolor(x2d, y2d, pressure[:,:], cmap=colormap)
  ax.set_title('pressure')
  fig.colorbar(plot, ax=ax)
  if nsims > 1:
    plt_fname = plt_dir_name+'/fig_'+f'{s:02d}'+'_pressure.png'
  else:
    plt_fname = plt_dir_name+'/fig_pressure.png'
  print('Saving %s' % plt_fname)
  plt.savefig(plt_fname)
  plt.close()

  fig = plt.figure(figsize=figsize)
  ax = plt.axes()
  ax.set( xlim=(np.min(x), np.max(x)), ylim=(0, t_final) )
  plot = ax.pcolor(x2d, y2d, temperature[:,:], cmap=colormap)
  ax.set_title('temperature')
  fig.colorbar(plot, ax=ax)
  if nsims > 1:
    plt_fname = plt_dir_name+'/fig_'+f'{s:02d}'+'_temperature.png'
  else:
    plt_fname = plt_dir_name+'/fig_temperature.png'
  print('Saving %s' % plt_fname)
  plt.savefig(plt_fname)
  plt.close()
