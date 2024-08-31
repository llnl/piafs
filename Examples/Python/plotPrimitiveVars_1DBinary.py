'''
Python script to create plots from the solution of a
PIAFS2D simulation.

- if op_overwrite is set to "no", a plot is generated
for each variable (solution vector component) and each
simulation time for which the solution is available.
- if op_overwrite is set to "yes", a single plot is
created for for each variable (solution vector component).
- solution must be in binary format

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

figsize=(15,9)
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
  solution_snapshots = np.float32(solution_snapshots)
  n_snapshots = solution_snapshots.shape[0]
  print('  number of snapshots = ', n_snapshots)

  for i in range(n_snapshots):
    for s in range(nsims):
        solution_snapshots_sim = solution_snapshots[s*n_snapshots:(s+1)*n_snapshots]
        cons_sol = np.transpose(solution_snapshots_sim.reshape(n_snapshots,size[0],nvars))[:,:,i]
        rho = cons_sol[0,:]
        u = cons_sol[1,:] / cons_sol[0,:]
        pressure = 0.4 * (cons_sol[2,:] - 0.5*cons_sol[1,:]*cons_sol[1,:]/cons_sol[0,:])
        temperature = pressure / rho

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.set( xlim=(np.min(x), np.max(x)),ylim=(np.min(rho),np.max(rho) ) )
        ax.plot(x, rho, lw=2)
        ax.set_title('rho, t={:.3}'.format(i*dt_snapshots))
        plt.grid(visible=True, linestyle=':', linewidth=1)
        plt_fname = plt_dir_name+'/fig_rho'+'_'+f'{i:05d}'+'.png'
        print('Saving %s' % plt_fname)
        plt.savefig(plt_fname)
        plt.close()

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.set( xlim=(np.min(x), np.max(x)),ylim=(np.min(u),np.max(u) ) )
        ax.plot(x, u, lw=2)
        ax.set_title('u, t={:.3}'.format(i*dt_snapshots))
        plt.grid(visible=True, linestyle=':', linewidth=1)
        plt_fname = plt_dir_name+'/fig_u'+'_'+f'{i:05d}'+'.png'
        print('Saving %s' % plt_fname)
        plt.savefig(plt_fname)
        plt.close()

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.set( xlim=(np.min(x), np.max(x)),ylim=(np.min(pressure),np.max(pressure) ) )
        ax.plot(x, pressure, lw=2)
        ax.set_title('pressure, t={:.3}'.format(i*dt_snapshots))
        plt.grid(visible=True, linestyle=':', linewidth=1)
        plt_fname = plt_dir_name+'/fig_pressure'+'_'+f'{i:05d}'+'.png'
        print('Saving %s' % plt_fname)
        plt.savefig(plt_fname)
        plt.close()

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.set( xlim=(np.min(x), np.max(x)),ylim=(np.min(temperature),np.max(temperature) ) )
        ax.plot(x, temperature, lw=2)
        ax.set_title('temperature, t={:.3}'.format(i*dt_snapshots))
        plt.grid(visible=True, linestyle=':', linewidth=1)
        plt_fname = plt_dir_name+'/fig_temperature'+'_'+f'{i:05d}'+'.png'
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
  print('  nvars = ', nvars)
  print('  grid size = ', size)
  print('  final time = ', t_final)
  print('  number of snapshots = ', n_snapshots)

  '''
  Load simulation data (solution snapshots)
  '''
  x,solution_snapshots = piafs2dutils.getSolutionSnapshots( sim_path,
                                                          nsims,
                                                          n_snapshots,
                                                          ndims,
                                                          nvars,
                                                          size )
  solution_snapshots = np.float32(solution_snapshots)

  for s in range(nsims):
      solution_snapshots_sim = solution_snapshots[s*n_snapshots:(s+1)*n_snapshots]
      cons_sol = np.transpose(solution_snapshots_sim.reshape(n_snapshots,size[0],nvars))[:,:]
      rho = cons_sol[0,:]
      u = cons_sol[1,:] / cons_sol[0,:]
      pressure = 0.4 * (cons_sol[2,:] - 0.5*cons_sol[1,:]*cons_sol[1,:]/cons_sol[0,:])
      temperature = pressure / rho

      fig = plt.figure(figsize=figsize)
      ax = plt.axes()
      ax.set( xlim=(np.min(x), np.max(x)),ylim=(np.min(rho),np.max(rho) ) )
      ax.plot(x, rho, lw=2)
      ax.set_title('rho, t={:.3}'.format(t_final))
      plt.grid(visible=True, linestyle=':', linewidth=1)
      plt_fname = plt_dir_name+'/fig_rho.png'
      print('Saving %s' % plt_fname)
      plt.savefig(plt_fname)
      plt.close()

      fig = plt.figure(figsize=figsize)
      ax = plt.axes()
      ax.set( xlim=(np.min(x), np.max(x)),ylim=(np.min(u),np.max(u) ) )
      ax.plot(x, u, lw=2)
      ax.set_title('u, t={:.3}'.format(t_final))
      plt.grid(visible=True, linestyle=':', linewidth=1)
      plt_fname = plt_dir_name+'/fig_u.png'
      print('Saving %s' % plt_fname)
      plt.savefig(plt_fname)
      plt.close()

      fig = plt.figure(figsize=figsize)
      ax = plt.axes()
      ax.set( xlim=(np.min(x), np.max(x)),ylim=(np.min(pressure),np.max(pressure) ) )
      ax.plot(x, pressure, lw=2)
      ax.set_title('pressure, t={:.3}'.format(t_final))
      plt.grid(visible=True, linestyle=':', linewidth=1)
      plt_fname = plt_dir_name+'/fig_pressure.png'
      print('Saving %s' % plt_fname)
      plt.savefig(plt_fname)
      plt.close()

      fig = plt.figure(figsize=figsize)
      ax = plt.axes()
      ax.set( xlim=(np.min(x), np.max(x)),ylim=(np.min(temperature),np.max(temperature) ) )
      ax.plot(x, temperature, lw=2)
      ax.set_title('temperature, t={:.3}'.format(t_final))
      plt.grid(visible=True, linestyle=':', linewidth=1)
      plt_fname = plt_dir_name+'/fig_temperature.png'
      print('Saving %s' % plt_fname)
      plt.savefig(plt_fname)
      plt.close()
