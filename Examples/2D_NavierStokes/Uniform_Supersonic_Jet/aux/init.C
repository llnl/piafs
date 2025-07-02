#include <fstream>
#include <iostream>
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
  double NA = 6.02214076e23; // Avogadro's number
  double kB = 1.380649e-23; // [J K^{-1}]; Boltzmann's constant
  double M_O2 = 0.032; // [kg]; O2 molar mass
  double M_CO2 = 0.044; // [kg]; O2 molar mass
  double Cp_O2 = 918.45; // [J kg^{-1} K^{-1}]; O2 Cp
  double Cv_O2 = 650.0; // [J kg^{-1} K^{-1}]; O2 Cv
  double Cp_CO2 = 842.86; // [J kg^{-1} K^{-1}]; CO2 Cp
  double Cv_CO2 = 654.5; // [J kg^{-1} K^{-1}]; CO2 Cv

  // Setup parameters

  double lambda_UV = 2.48e-7; // [m] (248 nm) - pump wavelength
  double theta = 0.17 * pi / 180; // radians; half angle between probe beams

  double f_CO2 = 0.0; // CO2 fraction
  double f_O3  = 0.05; // O3 fraction
  double Ptot = 101325.0; // [Pa]; total gas pressure
  double Ti = 288; // [K]; initial temperature

  double mu0_O2 = 1.99e-5; // @275K
  double kappa0_O2 = 2.70e-2; // @275K
  double mu0_CO2 = 1.45e-5; // @275K
  double kappa0_CO2 = 1.58e-2; // @275K
  double T0 = 275.0;
  double TS = 110.4;
  double TA = 245.4;
  double TB = 27.6;

  double Lz = 0.03; // 30 milimeters
  double z_mm = 0.0;
  int nz = 22;

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
        } else if (!strcmp(word,"Lz")) {
          fscanf(chem_in,"%lf",&Lz);
        } else if (!strcmp(word,"z_mm")) {
          fscanf(chem_in,"%lf",&z_mm);
        } else if (!strcmp(word,"nz")) {
          fscanf(chem_in,"%d",&nz);
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

  double kUV = 2 * pi / lambda_UV; //pump beam wave vector
  double kg = 2 * kUV * sin(theta); // grating wave vector

  double f_O2 = 1.0 - f_CO2; // O2 fraction

  double R = NA*kB/(f_O2*M_O2 + f_CO2*M_CO2); // Specific gas constant
  double Cp = f_O2*Cp_O2 + f_CO2*Cp_CO2;
  double Cv = f_O2*Cv_O2 + f_CO2*Cv_CO2;
  double gamma = Cp/Cv;

  double n_O2 = f_O2 * Ptot / (kB * Ti); // [m^{-3}]; initial O2 concentration
  double n_O3 = f_O3 * Ptot / (kB * Ti); // [m^{-3}]; initial O3 concentration
  double n_CO2 = f_CO2 * Ptot / (kB * Ti); // [m^{-3}]; initial CO2 concentration
  double rho_O2 = n_O2 * M_O2 / NA; // kg
  double rho_CO2 = n_CO2 * M_CO2 / NA; // kg
  double cs = sqrt(gamma*R*Ti); // [m s^{-1}]; speed of sound

  double mu_0 = f_O2*mu0_O2 + f_CO2*mu0_CO2;
  double kappa_0 = f_O2*kappa0_O2 + f_CO2*kappa0_CO2;
  double mu_ref = mu_0 * raiseto(Ti/T0, 1.5) * ((T0+TS)/(Ti+TS));
  double kappa_ref = kappa_0 * raiseto(Ti/T0, 1.5) * ((T0+TA*exp(-TB/T0))/(Ti+TA*exp(-TB/Ti)));

  // Normalization
  double L_ref = 1.0/kg;
  double T_ref = Ti;
  double rho_ref = Ptot/(R*T_ref);
  double v_ref = sqrt(gamma*R*T_ref); // [m s^{-1}]; speed of sound
  double t_ref = L_ref / v_ref;
  double P_ref = rho_ref * v_ref * v_ref;

  // Compute Reynolds and Prandtl numbers
  double Re = rho_ref * v_ref * L_ref / mu_ref;
  double Pr = gamma*R/(gamma-1) * mu_ref / kappa_ref;

  // compute z location
  double dz = Lz / nz;
  int z_i = (int)(ceil(z_mm*0.001 * nz / Lz));
  nz = z_i + 1;

  printf("Constants (in SI units):\n");
  printf("  gamma: %1.4e (m)\n", gamma);
  printf("  Avogradro number: %1.4e (m)\n", NA);
  printf("  Boltzmann constant: %1.4e (m)\n", kB);
  printf("  Specific gas constant: %1.4e (m)\n", R);
  printf("Photo-Chemistry:\n");
  printf("  Pump wavelength: %1.4e [m]\n", lambda_UV);
  printf("  Beam half angle: %1.4e [radians]\n", theta);
  printf("  Pump beam wavenumber: %1.4e [m^{-1}]\n", kUV);
  printf("  Grating wavenumber: %1.4e [m^{-1}]\n", kg);
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
  printf("  Gas length: %1.4e [m]\n", Lz);
  printf("  Z-location: %1.4e [m] (z_i = %d)\n", z_mm*1e-3,z_i);
  printf("  Number of z-layers: %d\n", nz);

  // domain
  double k_aw = 1; // normalized acoustic wave vector
  double l_aw = 2*pi/k_aw; // normalized acoustic wavelength
  double N_period_x = 5; // number of spatial periods
  double xmin = -N_period_x/2*l_aw;
  double xmax =  N_period_x/2*l_aw;
  printf("Physical domain:\n");
  printf("  normalized xmin, xmax: %1.4e, %1.4e\n", xmin, xmax);
  printf("  xmin/lambda_ac, xmax/lambda_ac: %1.4e, %1.4e\n", xmin/((2*pi/kg)/L_ref), xmax/((2*pi/kg)/L_ref));

  // uniform flow parameters (normalized) - vacuum
  double rho0 = (rho_O2 + rho_CO2) / rho_ref;
  double p0 = 1e7 / P_ref;
  double uvel0 = 0.0;
  double vvel0 = 1.63;
  printf("Initial flow: rho = %1.4e, p = %1.4e, (u, v) = (%1.4e, %1.4e)\n", rho0, p0, uvel0, vvel0);

  printf("\n");
	int NI, NJ, ndims, nvars;
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
        if (!strcmp(word, "ndims")) in >> ndims;
        else if (!strcmp(word, "nvars")) in >> nvars;
        else if (!strcmp(word, "size")) in >> NI >> NJ;
        else if (!strcmp(word, "ip_file_type")) in >> ip_file_type;
      }
    }else{
      std::cout << "Error: Illegal format in solver.inp. Crash and burn!\n";
    }
  }
  in.close();
  if (ndims != 2) {
    std::cout << "**ERROR**\n  ndims is not 2 in solver.inp. this code is to generate 2D exact solution\n";
    return(0);
  }
	std::cout << "Grid:" << NI << " X " << NJ << " points\n";

	int i,j;
	double dx = (xmax - xmin) / ((double)NI-1);
	double dy = dx;

  // allocate flow arrays
	double* x   = (double*) calloc (NI   , sizeof(double));
	double* y   = (double*) calloc (NJ   , sizeof(double));
	double* u0  = (double*) calloc (NI*NJ, sizeof(double));
	double* u1  = (double*) calloc (NI*NJ, sizeof(double));
	double* u2  = (double*) calloc (NI*NJ, sizeof(double));
	double* u3  = (double*) calloc (NI*NJ, sizeof(double));

  // set initial flow quantities
	for (i = 0; i < NI; i++){
  	for (j = 0; j < NJ; j++){

	  	x[i] = xmin + i*dx;
	  	y[j] = j*dy;

      int idx = NJ*i + j;

      u0[idx] = rho0;
      u1[idx] = rho0*uvel0;
      u2[idx] = rho0*vvel0;
      u3[idx] = p0/(gamma-1.0) + 0.5*rho0*(uvel0*uvel0+vvel0*vvel0);
	  }
	}

  // allocate reacting species arrays
  double* nv_O2    = (double*) calloc (NI*NJ*nz, sizeof(double));
  double* nv_O3    = (double*) calloc (NI*NJ*nz, sizeof(double));
  double* nv_1D    = (double*) calloc (NI*NJ*nz, sizeof(double));
  double* nv_1Dg   = (double*) calloc (NI*NJ*nz, sizeof(double));
  double* nv_3Su   = (double*) calloc (NI*NJ*nz, sizeof(double));
  double* nv_1Sg   = (double*) calloc (NI*NJ*nz, sizeof(double));
  double* nv_CO2   = (double*) calloc (NI*NJ*nz, sizeof(double));

  // set initial reacting species quantities
  for (i = 0; i < NI; i++){
  	for (j = 0; j < NJ; j++){
      int idx = NJ*i + j;
      int iz;
      for (iz = 0; iz < nz; iz++) {
        nv_O2 [nz*idx + iz] = 1.0;
        nv_O3 [nz*idx + iz] = n_O3 / n_O2;
        nv_1D [nz*idx + iz] = 0.0;
        nv_1Dg[nz*idx + iz] = 0.0;
        nv_3Su[nz*idx + iz] = 0.0;
        nv_1Sg[nz*idx + iz] = 0.0;
        nv_CO2[nz*idx + iz] = n_CO2 / n_O2;
      }
    }
  }

  // check
  const int n_species = 7; // O2, O3, 1D, 1Dg, 3Su, 1Sg, CO2
  const int n_ns_vars = 4;
  const int n_chem_vars = nz * n_species;
  if (nvars != n_ns_vars + n_chem_vars) {
    printf("\n**ERROR**\n  nvars in solver.inp is %d; it MUST be set to %d\n",
           nvars, n_ns_vars + n_chem_vars);
    return 1;
  }

  FILE *out;
  if (!strcmp(ip_file_type,"ascii")) {
    printf("Writing ASCII initial solution file initial.inp\n");
    out = fopen("initial.inp","w");
    // write x-coordinates
    for (i = 0; i < NI; i++)  fprintf(out,"%lf ",x[i]);
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)  fprintf(out,"%lf ",y[j]);
    fprintf(out,"\n");
    // write flow variables
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
    // write reacting species
    int iz;
    for (iz = 0; iz < nz; iz++) {
      for (j = 0; j < NJ; j++)  {
        for (i = 0; i < NI; i++)  {
          int p = NJ*i + j;
          fprintf(out,"%lf ",nv_O2[nz*p+iz]);
        }
      }
      fprintf(out,"\n");
      for (j = 0; j < NJ; j++)  {
        for (i = 0; i < NI; i++)  {
          int p = NJ*i + j;
          fprintf(out,"%lf ",nv_O3[nz*p+iz]);
        }
      }
      fprintf(out,"\n");
      for (j = 0; j < NJ; j++)  {
        for (i = 0; i < NI; i++)  {
          int p = NJ*i + j;
          fprintf(out,"%lf ",nv_1D[nz*p+iz]);
        }
      }
      fprintf(out,"\n");
      for (j = 0; j < NJ; j++)  {
        for (i = 0; i < NI; i++)  {
          int p = NJ*i + j;
          fprintf(out,"%lf ",nv_1Dg[nz*p+iz]);
        }
      }
      fprintf(out,"\n");
      for (j = 0; j < NJ; j++)  {
        for (i = 0; i < NI; i++)  {
          int p = NJ*i + j;
          fprintf(out,"%lf ",nv_3Su[nz*p+iz]);
        }
      }
      fprintf(out,"\n");
      for (j = 0; j < NJ; j++)  {
        for (i = 0; i < NI; i++)  {
          int p = NJ*i + j;
          fprintf(out,"%lf ",nv_1Sg[nz*p+iz]);
        }
      }
      fprintf(out,"\n");
      for (j = 0; j < NJ; j++)  {
        for (i = 0; i < NI; i++)  {
          int p = NJ*i + j;
          fprintf(out,"%lf ",nv_CO2[nz*p+iz]);
        }
      }
      fprintf(out,"\n");
    }
    fclose(out);

  } else if ((!strcmp(ip_file_type,"binary")) || (!strcmp(ip_file_type,"bin"))) {

    printf("Writing binary initial solution file initial.inp\n");
    out = fopen("initial.inp","wb");
    fwrite(x,sizeof(double),NI,out);
    fwrite(y,sizeof(double),NJ,out);
    double *U = (double*) calloc (nvars*NI*NJ,sizeof(double));
    for (i=0; i < NI; i++) {
      for (j = 0; j < NJ; j++) {
        int p = NJ*i + j;
        int q = NI*j + i;
        U[nvars*q+0] = u0[p];
        U[nvars*q+1] = u1[p];
        U[nvars*q+2] = u2[p];
        U[nvars*q+3] = u3[p];
        int iz;
        for (iz = 0; iz < nz; iz++) {
          U[nvars*q+n_ns_vars+iz*n_species+0] = nv_O2 [p*nz+iz];
          U[nvars*q+n_ns_vars+iz*n_species+1] = nv_O3 [p*nz+iz];
          U[nvars*q+n_ns_vars+iz*n_species+2] = nv_1D [p*nz+iz];
          U[nvars*q+n_ns_vars+iz*n_species+3] = nv_1Dg[p*nz+iz];
          U[nvars*q+n_ns_vars+iz*n_species+4] = nv_3Su[p*nz+iz];
          U[nvars*q+n_ns_vars+iz*n_species+5] = nv_1Sg[p*nz+iz];
          U[nvars*q+n_ns_vars+iz*n_species+6] = nv_CO2[p*nz+iz];
        }
      }
    }
    fwrite(U,sizeof(double),nvars*NI*NJ,out);
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

  free(nv_O2);
  free(nv_O3);
  free(nv_1D);
  free(nv_1Dg);
  free(nv_3Su);
  free(nv_1Sg);
  free(nv_CO2);

	return(0);
}
