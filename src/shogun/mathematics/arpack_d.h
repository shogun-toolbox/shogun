/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef ARPACK_D_H_
#define ARPACK_D_H_
#include <shogun/lib/config.h>
#ifdef HAVE_ARPACK
#ifdef HAVE_LAPACK
void arpack_dsxupd(double* matrix, double* rhs_diag, int n, int nev, const char* which,
                   int mode, bool pos, double shift, double tolerance,
                   double* eigenvalues, double* eigenvectors, int& status);
#endif /* HAVE_LAPACK */
#endif /* HAVE_ARPACK */
#endif /* ARPACK_D_H_ */

