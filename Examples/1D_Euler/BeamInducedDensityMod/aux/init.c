#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double raiseto(double x, double a)
{
  return exp(a*log(x));
}

int main()
{
  // Constants
  double pi = 4.0*atan(1.0); // pi
  double gamma = 1.4; // specific heat ratio
  double NA = 6.02214076e23; // Avogadro's number
  double kB = 1.380649e-23; // [J K^{-1}]; Boltzmann's constant
  double M_O2 = 0.032; // [kg]; O2 molar mass
  double M_CO2 = 0.044; // [kg]; O2 molar mass
  double R = NA*kB/M_O2; // Specific gas constant

  // Setup parameters

  double lambda_UV = 2.48e-7; // [m] (248 nm) - pump wavelength
  double theta = 0.17 * pi / 180; // radians; half angle between probe beams
  double kUV = 2 * pi / lambda_UV; //pump beam wave vector
  double kg = 2 * kUV * sin(theta); // grating wave vector

  double f_CO2 = 0.0; // CO2 fraction
  double f_O3  = 0.05; // O3 fraction
  double Ptot = 101325.0; // [Pa]; total gas pressure
  double Ti = 288; // [K]; initial temperature

  double mu_0 = 1.92e-5; // @275K
  double kappa_0 = 2.59e-2; // @275K
  double T0 = 275.0;
  double TS = 110.4;
  double TA = 245.4;
  double TB = 27.6;

  FILE *chem_in;
  printf("Reading chemical reaction inputs from file \"chemistry.inp\".\n");
  chem_in = fopen("chemistry.inp","r");
  if (!chem_in) printf("Warning: File \"chemistry.inp\" not found. Using default values.\n");
  else {
    char word[100];
    fscanf(chem_in,"%s",word);
    if (!strcmp(word, "begin")){
      while (strcmp(word, "end")){
        fscanf(chem_in,"%s",word);
        if (!strcmp(word,"lambda_UV")) {
          fscanf(chem_in,"%lf",&lambda_UV);
        } else if (!strcmp(word,"theta")) {
          fscanf(chem_in,"%lf",&theta);
        } else if (!strcmp(word,"f_CO2")) {
          fscanf(chem_in,"%lf",&f_CO2);
        } else if (!strcmp(word,"f_O3")) {
          fscanf(chem_in,"%lf",&f_O3);
        } else if (!strcmp(word,"Ptot")) {
          fscanf(chem_in,"%lf",&Ptot);
        } else if (!strcmp(word,"Ti")) {
          fscanf(chem_in,"%lf",&Ti);
        } else if (strcmp(word,"end")) {
          char useless[100];
          fscanf(chem_in,"%s",useless);
        }
      }
    } else {
      fprintf(stderr,"Error: Illegal format in file \"chemistry.inp\".\n");
      return(1);
    }
    fclose(chem_in);
  }

  /* sanity checks */
  if ((f_CO2 > 1.0) || (f_CO2 < 0.0)) {
    fprintf(stderr,"ERROR in ChemistryInitialize(): f_CO2 is not between 0.0 and 1.0 !!!\n");
    return 1;
  }
  if ((f_O3 > 1.0) || (f_O3 < 0.0)) {
    fprintf(stderr,"ERROR in ChemistryInitialize(): f_O3 is not between 0.0 and 1.0 !!!\n");
    return 1;
  }
  if ((f_CO2 + f_O3) > 1.0) {
    fprintf(stderr,"ERROR in ChemistryInitialize(): f_CO2 + f_O3 > 1.0 !!!\n");
    return 1;
  }

  double f_O2 = 1.0 - f_CO2 - f_O3; // O2 fraction
  double n_O2 = f_O2 * Ptot / (kB * Ti); // [m^{-3}]; initial O2 concentration
  double n_O3 = f_O3 * Ptot / (kB * Ti); // [m^{-3}]; initial O3 concentration
  double n_CO2 = f_CO2 * Ptot / (kB * Ti); // [m^{-3}]; initial CO2 concentration
  double rho_O2 = n_O2 * M_O2 / NA; // kg
  double rho_CO2 = n_CO2 * M_CO2 / NA; // kg
  double cs = sqrt(gamma*R*Ti); // [m s^{-1}]; speed of sound

  double mu_ref = mu_0 * raiseto(Ti/T0, 1.5) * ((T0+TS)/(Ti+TS));
  double kappa_ref = kappa_0 * raiseto(Ti/T0, 1.5) * ((T0+TA*exp(-TB/T0))/(Ti+TA*exp(-TB/Ti)));

  // Normalization
  double L_ref = 1.0/kg;
  double T_ref = Ti;
  double rho_ref = Ptot/(R*T_ref);
  double v_ref = cs;
  double t_ref = L_ref / v_ref;
  double P_ref = rho_ref * v_ref * v_ref;

  // Compute Reynolds and Prandtl numbers
  double Re = rho_ref * v_ref * L_ref / mu_ref;
  double Pr = gamma*R/(gamma-1) * mu_ref / kappa_ref;

  printf("Constants (in SI units):\n");
  printf("  gamma: %1.4e (m)\n", gamma);
  printf("  Avogradro number: %1.4e (m)\n", NA);
  printf("  Boltzmann constant: %1.4e (m)\n", kB);
  printf("  Specific gas constant: %1.4e (m)\n", R);
  printf("Reference quantities:\n");
  printf("  Length: %1.4e (m)\n", L_ref);
  printf("  Temperature: %1.4e (m)\n", T_ref);
  printf("  Density: %1.4e (kg m^{-3})\n", rho_ref);
  printf("  Speed: %1.4e (m s^{-1})\n", v_ref);
  printf("  Time: %1.4e (s)\n", t_ref);
  printf("  Pressure: %1.4e (Pa)\n", P_ref);
  printf("  Coeff. viscosity: %1.4e (kg s^{-1} m^{-1})\n", mu_ref);
  printf("  Coeff. conductivity: %1.4e (W m^{-1} K^{-1})\n", kappa_ref);
  printf("Other important quantities:\n");
  printf("  Speed of sound: %1.4e\n", cs);
  printf("  Reynolds number: %1.4e\n", Re);
  printf("  Prandtl number: %1.4e\n", Pr);
  printf("  lambda_ac (acoustic spatial period): %1.4e (m)\n", (2*pi/kg) );
  printf("    normalized lambda_ac: %1.4e\n", (2*pi/kg)/L_ref );
  printf("  tau_ac (acoustic time period): %1.4e (s)\n", (2*pi/kg)/cs );
  printf("    normalized tau_ac: %1.4e (s)\n", ((2*pi/kg)/cs)/t_ref );
  printf("Chemical species:\n");
  printf("  O2 fraction: %1.4e\n", f_O2);
  printf("  O3 fraction: %1.4e\n", f_O3);
  printf("  CO2 fraction: %1.4e\n", f_CO2);
  printf("  O2 number density:  %1.4e [m^{-3}]\n", n_O2);
  printf("  O3 number density:  %1.4e [m^{-3}]\n", n_O3);
  printf("  CO2 number density: %1.4e [m^{-3}]\n", n_CO2);
  printf("  O2 mass density: %1.4e [kg m^{-3}]\n", rho_O2);
  printf("  CO2 mass density: %1.4e [kg m^{-3}]\n", rho_CO2);

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
  double rho0 = (rho_O2 + rho_CO2) / rho_ref;
  double p0 = Ptot / P_ref;
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
