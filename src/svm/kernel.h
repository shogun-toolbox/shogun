/************************************************************************/
/*                                                                      */
/*   kernel.h                                                           */
/*                                                                      */
/*   User defined kernel function. Feel free to plug in your own.       */
/*                                                                      */
/*   Copyright: Thorsten Joachims                                       */
/*   Date: 16.12.97                                                     */
/*                                                                      */
/************************************************************************/

/* KERNEL_PARM is defined in svm_common.h The field 'custom' is reserved for */
/* parameters of the user defined kernel. You can also access and use */
/* the parameters of the other kernels. */

#ifndef _KERNEL_H___
#define _KERNEL_H___

#include "svm/svm_common.h"
#include "hmm/HMM.h"

void tester();
double find_normalizer(KERNEL_PARM *kernel_parm, int num);
double linear_top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b);
double top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b);

#endif
