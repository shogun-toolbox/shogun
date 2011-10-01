/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/lib/config.h>
#ifdef HAVE_ARPACK
#ifdef HAVE_LAPACK
#include <shogun/lib/common.h>
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/arpack_s.h>
#include <shogun/mathematics/arpack_d.h>

using namespace shogun;

namespace shogun
{

template <>
void arpack_xsxupd<float32_t>(float32_t* matrix, float32_t* rhs_diag, int n, int nev, const char* which,
                              int mode, bool pos, float32_t shift, float32_t tolerance,
                              float32_t* eigenvalues, float32_t* eigenvectors, int& status)
{
	arpack_ssxupd(matrix,rhs_diag,n,nev,which,mode,pos,shift,tolerance,eigenvalues,eigenvectors,status);
}

template <>
void arpack_xsxupd<float64_t>(float64_t* matrix, float64_t* rhs_diag, int n, int nev, const char* which,
                              int mode, bool pos, float64_t shift, float64_t tolerance,
                              float64_t* eigenvalues, float64_t* eigenvectors, int& status)
{
	arpack_dsxupd(matrix,rhs_diag,n,nev,which,mode,pos,shift,tolerance,eigenvalues,eigenvectors,status);
}
}
#endif
#endif
