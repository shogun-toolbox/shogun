#ifndef _LAPACK_H__
#define _LAPACK_H__

extern "C" {

void daxpy_(int, double, double*, int, double*, int) ;
double ddot_(int, double*, int, double*, int) ;

}

#endif
