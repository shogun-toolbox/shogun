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

#ifndef JADIAGORTH_H_
#define JADIAGORTH_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/mathematics/ajd/ApproxJointDiagonalizer.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

/** @brief Class JADiagOrth
 *
 * An Approximate Joint Diagonalizer (AJD) Implementation
 *
 * Cardoso, J. F., & Souloumiac, A. (1993).
 * Blind beamforming for non-Gaussian signals.
 * In IEE Proceedings F (Radar and Signal Processing)
 * (Vol. 140, No. 6, pp. 362-370). IET Digital Library.
 *
 */
class CJADiagOrth : public CApproxJointDiagonalizer
{
	public:

		/** constructor */
		CJADiagOrth()
		{
		}

		/** destructor */
		virtual ~CJADiagOrth()
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
		virtual const char* get_name() const { return "JADiagOrth"; }
};
}
#endif //HAVE_EIGEN3
#endif //JADIAGORTH_H_
