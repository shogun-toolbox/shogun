#ifndef _LAPACK_H__
#define _LAPACK_H__

extern "C" {

void daxpy_(int*, double*, double*, int*, double*, int*) ;
double ddot_(int*, double*, int*, double*, int*) ;
int dger_(int*, int*, double*, double*, int*, double*, int*, double*, int*) ;
int dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*) ;
int dgemv_(char *, int*, int*, double *, double *, int *, double *, int*, double*, double*, int*) ;

}

#endif
