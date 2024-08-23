#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main()
{
  // normalization
  // L_ref = 1e-6 meters (1 micrometer)
  // t_ref = 1e-9 seconds (1 nanosecond)
  // rho_ref = 1.225 kg m^{-3}
  // Consequently,
  // v_ref = L_ref / t_ref = 1e3 m/s
  // p_ref = rho_ref*v_ref^2 = 1.225e6 kg m^{-2}  `

  // domain
  double lambda_ac = 42.0; // 42 micrometers
  double xmin = -5*lambda_ac;
  double xmax = 5*lambda_ac;

  // uniform flow parameters
  double gamma = 1.4;
  double rho0 = 1.0;
  double p0 = 0.082714; // 101325.0 Pa / p_ref
  double u0 = 0.0;

  // pressure pertubation
  double width = 5.0; // 5 micrometers
  double ampl = 1.0*p0;

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
  printf("Grid:\t\t\t%d\n",NI);

  int i;
  double dx = (xmax - xmin) / ((double)(NI-1));

  double *x, *rho,*rhou,*e;
  x    = (double*) calloc (NI, sizeof(double));
  rho  = (double*) calloc (NI, sizeof(double));
  rhou = (double*) calloc (NI, sizeof(double));
  e    = (double*) calloc (NI, sizeof(double));

  for (i = 0; i < NI; i++){
    x[i] = xmin + i*dx;
    double sigma = width / (2.0 * sqrt(2.0*log(2.0)));
    double dp = ampl*exp(-0.5*(x[i]/sigma)*(x[i]/sigma));
    double pressure = p0 + dp;
    double energy = pressure/(gamma-1.0) + 0.5*rho0*u0*u0;
    rho[i]  = rho0;
    rhou[i] = rho0 * u0;
    e[i]    = energy;
  }

  if (!strcmp(ip_file_type,"ascii")) {
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

  free(x);
  free(rho);
  free(rhou);
  free(e);

  return(0);
}
