#include "lib/common.h"

void nrerror2(char error_text[]);
REAL *vector(long nl, long nh);
int *ivector(long nl, long nh) ;
unsigned char *cvector(long nl, long nh) ;
unsigned long *lvector(long nl, long nh) ;
double *dvector(long nl, long nh) ;
REAL **matrix(long nrl, long nrh, long ncl, long nch) ;
double **dmatrix(long nrl, long nrh, long ncl, long nch) ;
int **imatrix(long nrl, long nrh, long ncl, long nch) ;
REAL **convert_matrix(REAL *a, long nrl, long nrh, long ncl, long nch) ;
REAL ***f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh) ;
void free_vector(REAL *v, long nl, long nh) ;
void free_ivector(int *v, long nl, long nh) ;
void free_cvector(unsigned char *v, long nl, long nh) ;
void free_lvector(unsigned long *v, long nl, long nh) ;
void free_dvector(double *v, long nl, long nh) ;
void free_matrix(REAL **m, long nrl, long nrh, long ncl, long nch) ;
void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch) ;
void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch) ;
void free_submatrix(REAL **b, long nrl, long nrh, long ncl, long nch) ;
void free_convert_matrix(REAL **b, long nrl, long nrh, long ncl, long nch) ;
void free_f3tensor(REAL ***t, long nrl, long nrh, long ncl, long nch, 
		   long ndl, long ndh) ;

inline double sign(double tol, double a)
{
  if (a<0) return -tol ;
  if (a>0) return tol ;
  return 0 ;
} ;

inline REAL fmax(REAL a, REAL b)
{
  if (a>b)
    return a ;
  return b ;
} ;
