/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef ARPACK_S_H_
#define ARPACK_S_H_
#include <shogun/lib/config.h>
#ifdef HAVE_ARPACK
#ifdef HAVE_LAPACK
void arpack_ssxupd(float* matrix, float* rhs_diag, int n, int nev, const char* which,
                   int mode, bool pos, float shift, float tolerance,
                   float* eigenvalues, float* eigenvectors, int& status);
#endif /* HAVE_LAPACK */
#endif /* HAVE_ARPACK */
#endif /* ARPACK_S_H_ */

