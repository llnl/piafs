#include <fstream>
#include <iostream>
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

  const double f_CO2 = 0.04; // CO2 fraction
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
  double N_period_x = 10; // number of spatial periods
  double xmin = -N_period_x/2*l_aw;
  double xmax =  N_period_x/2*l_aw;
  printf("Physical domain:\n");
  printf("  normalized xmin, xmax: %1.4e, %1.4e\n", xmin, xmax);
  printf("  xmin/lambda_ac, xmax/lambda_ac: %1.4e, %1.4e\n", xmin/((2*pi/kg)/L_ref), xmax/((2*pi/kg)/L_ref));

  // uniform flow parameters
  double rho0 = 1.0;
  double p0 = 1.0;
  double uvel0 = 0.0;
  double vvel0 = 0.0;
  printf("Initial flow: rho = %1.4e, p = %1.4e, (u, v) = (%1.4e, %1.4e)\n", rho0, p0, uvel0, vvel0);

  // pressure pertubation
  double width = 5.0;
  double ampl = 1.0*p0;
  printf("Pressure perturbation: width = %1.2e, amplitude = %1.2e\n", width, ampl);

  printf("\n");
	int NI, NJ, ndims;
  char ip_file_type[50]; strcpy(ip_file_type,"ascii");

  std::ifstream in;
  std::cout << "Reading file \"solver.inp\"...\n";
  in.open("solver.inp");
  if (!in) {
    std::cout << "Error: Input file \"solver.inp\" not found. Default values will be used.\n";
  } else {
    char word[500];
    in >> word;
    if (!strcmp(word, "begin")){
      while (strcmp(word, "end")){
        in >> word;
        if (!strcmp(word, "ndims"))     in >> ndims;
        else if (!strcmp(word, "size")) in >> NI >> NJ;
        else if (!strcmp(word, "ip_file_type")) in >> ip_file_type;
      }
    }else{
      std::cout << "Error: Illegal format in solver.inp. Crash and burn!\n";
    }
  }
  in.close();
  if (ndims != 2) {
    std::cout << "ndims is not 2 in solver.inp. this code is to generate 2D exact solution\n";
    return(0);
  }
	std::cout << "Grid:" << NI << " X " << NJ << " points\n";

	int i,j;
	double dx = (xmax - xmin) / ((double)NI-1);
	double dy = dx;

	double *x, *y, *u0, *u1, *u2, *u3;
  FILE *out;

	x   = (double*) calloc (NI   , sizeof(double));
	y   = (double*) calloc (NJ   , sizeof(double));
	u0  = (double*) calloc (NI*NJ, sizeof(double));
	u1  = (double*) calloc (NI*NJ, sizeof(double));
	u2  = (double*) calloc (NI*NJ, sizeof(double));
	u3  = (double*) calloc (NI*NJ, sizeof(double));

	for (i = 0; i < NI; i++){
  	for (j = 0; j < NJ; j++){

	  	x[i] = xmin + i*dx;
	  	y[j] = j*dy;

      double sigma = width / (2.0 * sqrt(2.0*log(2.0)));
      double dp = ampl*exp(-0.5*(x[i]/sigma)*(x[i]/sigma));
      double pressure = p0 + dp;

      int idx = NJ*i + j;

      u0[idx] = rho0;
      u1[idx] = rho0*uvel0;
      u2[idx] = rho0*vvel0;
      u3[idx] = pressure/(gamma-1.0) + 0.5*rho0*(uvel0*uvel0+vvel0*vvel0);
	  }
	}

  if (!strcmp(ip_file_type,"ascii")) {
    printf("Writing ASCII initial solution file initial.inp\n");
    out = fopen("initial.inp","w");
    for (i = 0; i < NI; i++)  fprintf(out,"%lf ",x[i]);
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)  fprintf(out,"%lf ",y[j]);
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)  {
      for (i = 0; i < NI; i++)  {
        int p = NJ*i + j;
        fprintf(out,"%lf ",u0[p]);
      }
    }
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)  {
      for (i = 0; i < NI; i++)  {
        int p = NJ*i + j;
        fprintf(out,"%lf ",u1[p]);
      }
    }
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)  {
      for (i = 0; i < NI; i++)  {
        int p = NJ*i + j;
        fprintf(out,"%lf ",u2[p]);
      }
    }
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)  {
      for (i = 0; i < NI; i++)  {
        int p = NJ*i + j;
        fprintf(out,"%lf ",u3[p]);
      }
    }
    fprintf(out,"\n");
    fclose(out);

  } else if ((!strcmp(ip_file_type,"binary")) || (!strcmp(ip_file_type,"bin"))) {

    printf("Writing binary initial solution file initial.inp\n");
    out = fopen("initial.inp","wb");
    fwrite(x,sizeof(double),NI,out);
    fwrite(y,sizeof(double),NJ,out);
    double *U = (double*) calloc (4*NI*NJ,sizeof(double));
    for (i=0; i < NI; i++) {
      for (j = 0; j < NJ; j++) {
        int p = NJ*i + j;
        int q = NI*j + i;
        U[4*q+0] = u0[p];
        U[4*q+1] = u1[p];
        U[4*q+2] = u2[p];
        U[4*q+3] = u3[p];
      }
    }
    fwrite(U,sizeof(double),4*NI*NJ,out);
    free(U);
    fclose(out);
  }
  printf("Done!\n");

	free(x);
	free(y);
	free(u0);
	free(u1);
	free(u2);
	free(u3);

	return(0);
}
