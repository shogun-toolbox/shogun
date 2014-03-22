/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 pl8787
 * Written (W) 2014 Wu Lin
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 *
 * ALGLIB Copyright 1984, 1987, 1995, 2000 by Stephen L. Moshier under GPL2+
 * http://www.alglib.net/
 * See method comments which functions are taken from ALGLIB (with adjustments
 * for shogun)
 */

#include <mathematics/Elmat.h>

using namespace shogun;

#ifdef HAVE_EIGEN3
using namespace Eigen;
#if 0
template<typename M1>
MatrixXd CElmat::norm_pdf(const MatrixBase<M1> &x)
{

}

template<typename M1>
MatrixXd CElmat::norm_cdf(const MatrixBase<M1> &x)
{

}

template<typename M1, typename M2>
MatrixXd bsxfun(CElmat::BsxfunOp op,const MatrixBase<M1> &x,const MatrixBase<M2> &y)
{

}
#endif
#endif //HAVE_EIGEN3
