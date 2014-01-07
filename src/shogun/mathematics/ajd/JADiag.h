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

#ifndef JADIAG_H_
#define JADIAG_H_

#include <lib/config.h>

#ifdef HAVE_EIGEN3

#include <mathematics/ajd/ApproxJointDiagonalizer.h>
#include <mathematics/Math.h>

namespace shogun
{

/** @brief Class JADiag
 *
 * An Approximate Joint Diagonalizer (AJD) Implementation
 * Assumes the matrices are Positive-definite
 *
 * Pham, D. T., & Cardoso, J. F. (2001).
 * Blind separation of instantaneous mixtures of nonstationary sources.
 * Signal Processing, IEEE Transactions on, 49(9), 1837-1848.
 *
 */
class CJADiag : public CApproxJointDiagonalizer
{
	public:

		/** constructor */
		CJADiag()
		{
		}

		/** destructor */
		virtual ~CJADiag()
		{
		}

		/** Computes the matrix V that best diagonalizes C
		 * @param C the set of matrices to be diagonalized
		 * @param V0 an estimate of the matrix V
		 * @param eps machine epsilon or desired epsilon
		 * @param itermax maximum number of iterations
		 * @return V the matrix that best diagonalizes C
		 */
		static SGMatrix<float64_t> diagonalize(SGNDArray<float64_t> C,
							SGMatrix<float64_t> V0 = SGMatrix<float64_t>(NULL,0,0,false),
							double eps=CMath::MACHINE_EPSILON,
							int itermax=200);

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
						   int itermax=200)
		{
			m_V = diagonalize(C,V0,eps,itermax);
			return m_V;
		}

		/** @return object name */
		virtual const char* get_name() const { return "JADiag"; }
};
}
#endif //HAVE_EIGEN3
#endif //JADIAG_H_
