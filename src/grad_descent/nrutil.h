#include "lib/common.h"

void nrerror2(CHAR error_text[]);
REAL *vector(LONG nl, LONG nh);
int *ivector(LONG nl, LONG nh) ;
BYTE *cvector(LONG nl, LONG nh) ;
unsigned long *lvector(LONG nl, LONG nh) ;
double *dvector(LONG nl, LONG nh) ;
REAL **matrix(LONG nrl, LONG nrh, LONG ncl, LONG nch) ;
double **dmatrix(LONG nrl, LONG nrh, LONG ncl, LONG nch) ;
int **imatrix(LONG nrl, LONG nrh, LONG ncl, LONG nch) ;
REAL **convert_matrix(REAL *a, LONG nrl, LONG nrh, LONG ncl, LONG nch) ;
REAL ***f3tensor(LONG nrl, LONG nrh, LONG ncl, LONG nch, LONG ndl, LONG ndh) ;
void free_vector(REAL *v, LONG nl, LONG nh) ;
void free_ivector(int *v, LONG nl, LONG nh) ;
void free_cvector(BYTE *v, LONG nl, LONG nh) ;
void free_lvector(unsigned long *v, LONG nl, LONG nh) ;
void free_dvector(double *v, LONG nl, LONG nh) ;
void free_matrix(REAL **m, LONG nrl, LONG nrh, LONG ncl, LONG nch) ;
void free_dmatrix(double **m, LONG nrl, LONG nrh, LONG ncl, LONG nch) ;
void free_imatrix(int **m, LONG nrl, LONG nrh, LONG ncl, LONG nch) ;
void free_submatrix(REAL **b, LONG nrl, LONG nrh, LONG ncl, LONG nch) ;
void free_convert_matrix(REAL **b, LONG nrl, LONG nrh, LONG ncl, LONG nch) ;
void free_f3tensor(REAL ***t, LONG nrl, LONG nrh, LONG ncl, LONG nch, 
		   LONG ndl, LONG ndh) ;

inline double sign(double tol, double a)
{
  if (a<0) return -tol ;
  if (a>0) return tol ;
  return 0 ;
} ;
