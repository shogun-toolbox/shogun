/************************************************************************/
/*                                                                      */
/*   kernel.h                                                           */
/*                                                                      */
/*   User defined kernel function. Feel free to plug in your own.       */
/*                                                                      */
/*                                                                      */
/************************************************************************/

/* KERNEL_PARM is defined in svm_common.h The field 'custom' is reserved for */
/* parameters of the user defined kernel. You can also access and use */
/* the parameters of the other kernels. */

//#ifndef _KERNEL_H___
//#define _KERNEL_H___

#include "hmm/HMM.h"
#include "lib/Observation.h"
#include "lib/common.h"

typedef struct kernel_parm {
  long    kernel_type;   /* 0=linear, 1=poly, 2=rbf, 3=sigmoid, 4=custom */
  long    poly_degree;
  double  rbf_gamma;
  double  coef_lin;
  double  coef_const;
  char    custom[50];    /* for user supplied kernel */
} KERNEL_PARM;

void tester(KERNEL_PARM *kernel_parm);
double find_normalizer(KERNEL_PARM *kernel_parm, int num);
double linear_top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b);
double top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b);
double cached_top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b);
CFLOAT kernel(KERNEL_PARM *, DOC *, DOC *); 

//#endif
