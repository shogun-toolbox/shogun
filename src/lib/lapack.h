#ifndef _LAPACK_H__
#define _LAPACK_H__

#include "lib/common.h"

extern "C" {

void daxpy_(int*, double*, double*, int*, double*, int*) ;
double ddot_(int*, double*, int*, double*, int*) ;
INT dger_(int*, int*, double*, double*, int*, double*, int*, double*, int*) ;
INT dsyev_(CHAR*, CHAR*, int*, double*, int*, double*, double*, int*, int*) ;
INT dgemv_(CHAR *, int*, int*, double *, double *, INT *, double *, int*, double*, double*, int*) ;

}

#endif
