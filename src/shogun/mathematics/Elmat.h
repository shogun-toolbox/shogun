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

#ifndef ELMAT_H_
#define ELMAT_H_

#include <shogun/lib/config.h>

#include <math.h>
#include <shogun/base/SGObject.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
using namespace Eigen;
#endif //HAVE_EIGEN3

namespace shogun
{

class CElmat: public CSGObject
{

public:

#ifdef HAVE_EIGEN3

	/** Eigen version of element-wise matrix norm_pdf.
	 *
	 * Returns a MatrixXd value with every element in x do
	 * Statistics::norm_pdf
	 *
	 * Added by pl8787
	 */
	template<typename M1>
	static MatrixXd norm_pdf(const MatrixBase<M1> &x, float64_t std_dev=1);

	/** Eigen version of element-wise matrix norm_cdf.
	 *
	 * Returns a MatrixXd value with every element in x do
	 * Statistics::norm_cdf
	 *
	 * Added by pl8787
	 */
	template<typename M1>
	static MatrixXd norm_cdf(const MatrixBase<M1> &x, float64_t std_dev=1);

	/** Type of Bsxfun Operation
	 *
	 * plus    Plus
	 * minus   Minus
	 * times   Array multiply
	 *
	 * Added by pl8787
	 */
	typedef enum _bsxfunOp
	{
		plus,
		minus,
		times
	} BsxfunOp;

	/** Binary Singleto Expansion Function
	 *
	 * Return the element-by-element binary operation,
	 * with singleton expansion enabled. op can be one
	 * of the following BsxfunOp value.
	 *
	 * plus    Plus
	 * minus   Minus
	 * times   Array multiply
	 *
	 * Added by pl8787, Wu Lin
	 */
	template<typename M1, typename M2>
	static MatrixXd bsxfun(BsxfunOp op,const MatrixBase<M1> &x,const MatrixBase<M2> &y);

#endif //HAVE_EIGEN3

};

#endif /* ELMAT_H_ */
