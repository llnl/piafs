#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main()
{
  double pi = 4.0*atan(1.0); // pi

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

  double a = 1.0;
  double b = 1.0;
  double c = 0.0;

  // domain
  double k_aw = 1; // normalized acoustic wave vector
  double l_aw = 2*pi/k_aw; // normalized acoustic wavelength
  double N_period_x = 5; // number of spatial periods
  double xmin = -N_period_x/2*l_aw;
  double xmax =  N_period_x/2*l_aw;
  printf("Physical domain:\n");
  printf("  normalized xmin, xmax: %1.4e, %1.4e\n", xmin, xmax);

	int i,j;
	double dx = (xmax-xmin) / ((double)NI-1);
	double dy = dx;

	double *x, *y, *ival;
  FILE *out;

	x    = (double*) calloc (NI   , sizeof(double));
	y    = (double*) calloc (NJ   , sizeof(double));
	ival = (double*) calloc (NI*NJ, sizeof(double));

	for (i = 0; i < NI; i++){
  	for (j = 0; j < NJ; j++){

	  	x[i] = xmin+i*dx;
	  	y[j] = j*dy;

      int idx = NI*j + i;
      double xt = x[i] - xmin;
      ival[idx] = a + b * cos(xt*(1-c*xt));
	  }
	}

  if (!strcmp(ip_file_type,"ascii")) {
    printf("Writing ASCII intensity map file imap.inp\n");
    out = fopen("imap.inp","w");
    for (i = 0; i < NI; i++)  fprintf(out,"%lf ",x[i]);
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)  fprintf(out,"%lf ",y[j]);
    fprintf(out,"\n");
    for (j = 0; j < NJ; j++)  {
      for (i = 0; i < NI; i++)  {
        int p = NI*j + i;
        fprintf(out,"%lf ",ival[p]);
      }
    }
    fprintf(out,"\n");
    fclose(out);

  } else if ((!strcmp(ip_file_type,"binary")) || (!strcmp(ip_file_type,"bin"))) {

    printf("Writing binary intensity map file imap.inp\n");
    out = fopen("imap.inp","wb");
    fwrite(x,sizeof(double),NI,out);
    fwrite(y,sizeof(double),NJ,out);
    fwrite(ival,sizeof(double),NI*NJ,out);
    fclose(out);
  }
  printf("Done!\n");

	free(x);
	free(y);
	free(ival);

	return(0);
}
