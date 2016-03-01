/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 *
 * Thanks to Andreas Ziehe and Cedric Gouy-Pailler
 */

#ifndef APPROXJOINTDIAGONALIZER_H_
#define APPROXJOINTDIAGONALIZER_H_

#include <shogun/lib/config.h>


#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGNDArray.h>
#include <shogun/base/SGObject.h>

#include <shogun/mathematics/Math.h>

namespace shogun
{

/** @brief Class ApproxJointDiagonalizer defines an
 * Approximate Joint Diagonalizer (AJD) interface.
 *
 * AJD finds the matrix V that best diagonalizes
 * a set \f${C^1 ... C^k}\f$ of real valued symmetric
 * \f$NxN\f$ matrices - \f$V*C*V^T\f$
 */
class CApproxJointDiagonalizer : public CSGObject
{
	public:

		/** constructor */
		CApproxJointDiagonalizer() : CSGObject()
		{
		};

		/** destructor */
		virtual ~CApproxJointDiagonalizer()
		{
		}

		/** Computes the matrix V that best diagonalizes C
		 * @param C the set of matrices to be diagonalized
		 * @param V0 an estimate of the matrix V
		 * @param eps machine epsilon or desired epsilon
		 * @param itermax maximum number of iterations
		 * @return V the matrix that best diagonalizes C
		 */
		virtual SGMatrix<float64_t> compute(SGNDArray<float64_t> C,
						   SGMatrix<float64_t> V0 = SGMatrix<float64_t>(NULL,0,0,false),
						   double eps=CMath::MACHINE_EPSILON,
						   int itermax=200) = 0;

		/** return the matrix V that best diagonalizes C */
		SGMatrix<float64_t> get_V()
		{
			return m_V;
		}

	protected:

		/** the matrix V that best diagonalizes C */
		SGMatrix<float64_t> m_V;

};
}
#endif //APPROXJOINTDIAGONALIZER_H_
