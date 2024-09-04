#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main()
{
  // Constants
  const double pi = 4.0*atan(1.0); // pi
  const double gamma = 1.4; // specific heat ratio
  const double NA = 6.02214076e23; // Avogadro's number
  const double kB = 1.380649e-23; // [J K^{-1}]; Boltzmann's constant

  // Setup parameters

  const double lambda_UV = 2.48e-7; // [m] (248 nm) - pump wavelength
  const double theta = 0.17 * pi / 180; // radians; half angle between probe beams
  const double kUV = 2 * pi / lambda_UV; //pump beam wave vector
  const double kg = 2 * kUV * sin(theta); // grating wave vector

  const double f_CO2 = 0.0; // CO2 fraction
  const double f_O2 = 1.0 - f_CO2; // O2 fraction
  const double Ptot = 101325.0; // [Pa]; total gas pressure
  const double Ti = 288; // [K]; initial temperature
  const double M_O2 = 0.032; // [kg]; O2 molar mass
  const double n_O2 = f_O2 * Ptot / (kB * Ti); // [m^{-3}]; initial O2 concentration
  const double rho_O2 = n_O2 * M_O2 / NA; // kg
  const double cs = sqrt(gamma*Ptot/rho_O2); // [m s^{-1}]; speed of sound

  // Normalization
  const double L_ref = 1.0/kg;
  const double v_ref = cs;
  const double t_ref = L_ref / v_ref;
  const double rho_ref = rho_O2;
  const double P_ref = rho_O2 * v_ref * v_ref;
  printf("Reference quantities:\n");
  printf("  Length: %1.4e (m)\n", L_ref);
  printf("  Time: %1.4e (s)\n", t_ref);
  printf("  Speed: %1.4e (m s^{-1})\n", v_ref);
  printf("  Density: %1.4e (kg m^{-3})\n", rho_ref);
  printf("  Pressure: %1.4e (Pa)\n", P_ref);
  printf("Other important quantities:\n");
  printf("  lambda_ac (acoustic spatial period): %1.4e (m)\n", (2*pi/kg) );
  printf("    normalized lambda_ac: %1.4e\n", (2*pi/kg)/L_ref );
  printf("  tau_ac (acoustic time period): %1.4e (s)\n", (2*pi/kg)/cs );
  printf("    normalized tau_ac: %1.4e (s)\n", ((2*pi/kg)/cs)/t_ref );

  // domain
  double k_aw = 1; // normalized acoustic wave vector
  double l_aw = 2*pi/k_aw; // normalized acoustic wavelength
  double N_period_x = 5; // number of spatial periods
  double xmin = -N_period_x/2*l_aw;
  double xmax =  N_period_x/2*l_aw;
  printf("Physical domain:\n");
  printf("  normalized xmin, xmax: %1.4e, %1.4e\n", xmin, xmax);
  printf("  xmin/lambda_ac, xmax/lambda_ac: %1.4e, %1.4e\n", xmin/((2*pi/kg)/L_ref), xmax/((2*pi/kg)/L_ref));

  // uniform flow parameters
  double rho0 = 1.0;
  double p0 = 1.0;
  double u0 = 0.0;
  printf("Initial flow: rho = %1.4e, p = %1.4e, u = %1.4e\n", rho0, p0, u0);

  printf("\n");
  int NI, ndims;
  // default values
  char ip_file_type[50]; strcpy(ip_file_type,"ascii");

  FILE *in;
  printf("Reading file \"solver.inp\"...\n");
  in = fopen("solver.inp","r");
  if (!in) {
    printf("Error: Input file \"solver.inp\" not found. Default values will be used.\n");
  } else {
    char word[500];
    fscanf(in,"%s",word);
    if (!strcmp(word, "begin")){
      while (strcmp(word, "end")){
        fscanf(in,"%s",word);
        if (!strcmp(word, "ndims"))     fscanf(in,"%d",&ndims);
        else if (!strcmp(word, "size")) fscanf(in,"%d",&NI);
        else if (!strcmp(word, "ip_file_type")) fscanf(in,"%s",ip_file_type);
      }
    } else printf("Error: Illegal format in solver.inp. Crash and burn!\n");
  }
  fclose(in);

  if (ndims != 1) {
    printf("ndims is not 1 in solver.inp. this code is to generate 1D initial conditions\n");
    return(0);
  }
  printf("Grid: %d points\n",NI);

  int i;
  double dx = (xmax - xmin) / ((double)(NI-1));

  double *x, *rho,*rhou,*e;
  x    = (double*) calloc (NI, sizeof(double));
  rho  = (double*) calloc (NI, sizeof(double));
  rhou = (double*) calloc (NI, sizeof(double));
  e    = (double*) calloc (NI, sizeof(double));

  for (i = 0; i < NI; i++){
    x[i] = xmin + i*dx;
    double energy = p0/(gamma-1.0) + 0.5*rho0*u0*u0;
    rho[i]  = rho0;
    rhou[i] = rho0 * u0;
    e[i]    = energy;
  }

  if (!strcmp(ip_file_type,"ascii")) {
    printf("Writing ascii initial solution file initial.inp\n");
    FILE *out;
    out = fopen("initial.inp","w");
    for (i = 0; i < NI; i++)  fprintf(out,"%lf ",x[i]);
    fprintf(out,"\n");
    for (i = 0; i < NI; i++)  fprintf(out,"%lf ",rho[i]);
    fprintf(out,"\n");
    for (i = 0; i < NI; i++)  fprintf(out,"%lf ",rhou[i]);
    fprintf(out,"\n");
    for (i = 0; i < NI; i++)  fprintf(out,"%lf ",e[i]);
    fprintf(out,"\n");
    fclose(out);
  } else if ((!strcmp(ip_file_type,"binary")) || (!strcmp(ip_file_type,"bin"))) {
    printf("Error: Writing binary initial solution file not implemented. ");
    printf("Please choose ip_file_type in solver.inp as \"ascii\".\n");
  }
  printf("Done!\n");

  free(x);
  free(rho);
  free(rhou);
  free(e);

  return(0);
}
