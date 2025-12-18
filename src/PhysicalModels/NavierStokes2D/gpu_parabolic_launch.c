/*! @file gpu_parabolic_launch.c
    @brief GPU parabolic kernel launch wrappers (CPU fallback)
*/

#include <gpu.h>
#ifndef GPU_NONE
#include <gpu_parabolic.h>
#endif

#define DEFAULT_BLOCK_SIZE 256

void gpu_launch_ns2d_get_primitive(double *Q, const double *u, int nvars, int npoints, double gamma, int blockSize)
{
#ifdef GPU_NONE
  (void)Q; (void)u; (void)nvars; (void)npoints; (void)gamma; (void)blockSize;
#else
  (void)Q; (void)u; (void)nvars; (void)npoints; (void)gamma; (void)blockSize;
#endif
}

void gpu_launch_ns2d_viscous_flux_x(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr, int blockSize
)
{
#ifdef GPU_NONE
  (void)FViscous; (void)Q; (void)QDerivX; (void)QDerivY;
  (void)nvars; (void)npoints; (void)Tref; (void)T0; (void)TS; (void)TA; (void)TB;
  (void)inv_Re; (void)inv_gamma_m1; (void)inv_Pr; (void)blockSize;
#else
  (void)FViscous; (void)Q; (void)QDerivX; (void)QDerivY;
  (void)nvars; (void)npoints; (void)Tref; (void)T0; (void)TS; (void)TA; (void)TB;
  (void)inv_Re; (void)inv_gamma_m1; (void)inv_Pr; (void)blockSize;
#endif
}

void gpu_launch_ns2d_viscous_flux_y(
  double *FViscous, const double *Q, const double *QDerivX, const double *QDerivY,
  int nvars, int npoints, double Tref, double T0, double TS, double TA, double TB,
  double inv_Re, double inv_gamma_m1, double inv_Pr, int blockSize
)
{
#ifdef GPU_NONE
  (void)FViscous; (void)Q; (void)QDerivX; (void)QDerivY;
  (void)nvars; (void)npoints; (void)Tref; (void)T0; (void)TS; (void)TA; (void)TB;
  (void)inv_Re; (void)inv_gamma_m1; (void)inv_Pr; (void)blockSize;
#else
  (void)FViscous; (void)Q; (void)QDerivX; (void)QDerivY;
  (void)nvars; (void)npoints; (void)Tref; (void)T0; (void)TS; (void)TA; (void)TB;
  (void)inv_Re; (void)inv_gamma_m1; (void)inv_Pr; (void)blockSize;
#endif
}

