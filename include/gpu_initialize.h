/*! @file gpu_initialize.h
    @brief GPU initialization function declarations
*/

#ifndef _GPU_INITIALIZE_H_
#define _GPU_INITIALIZE_H_

#include <simulation_object.h>

/* GPU memory allocation functions */
int GPUAllocateSolutionArrays(SimulationObject *simobj, int nsims);
int GPUAllocateGridArrays(SimulationObject *simobj, int nsims);
int GPUCopyGridArraysToDevice(SimulationObject *simobj, int nsims);
void GPUFreeSolutionArrays(SimulationObject *simobj, int nsims);

#endif /* _GPU_INITIALIZE_H_ */

