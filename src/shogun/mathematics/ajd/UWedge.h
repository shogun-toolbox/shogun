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

#ifndef UWEDGE_H_
#define UWEDGE_H_

#ifdef HAVE_EIGEN3

#include <shogun/mathematics/ajd/ApproxJointDiagonalizer.h>

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGNDArray.h>

#include <shogun/mathematics/Math.h>

namespace shogun
{

/** @brief Class FFDiag
 *
 * An Approximate Joint Diagonalizer (AJD) Implementation
 */
class CUWedge : public CApproxJointDiagonalizer
{
	public:
	
		/** constructor */
		CUWedge()
		{
		};

		/** destructor */
		virtual ~CUWedge()
		{
		}
	
		/** Computes the matrix V that best diagonalizes C 
		 * @param C the set of matrices to be diagonalized
		 * @param V0 an estimate of the matrix V
		 * @param eps machine epsilon or desired epsilon
		 * @param itermax maximum number of iterations
		 * @return V the matrix the best diagonalizes C 
		 */
		static SGMatrix<float64_t> diagonalize(SGNDArray<float64_t> &C,
       							SGMatrix<float64_t> V0= SGMatrix<float64_t>(NULL,0,0),
							double eps=CMath::MACHINE_EPSILON,
							int itermax=200);

		/** Computes the matrix V that best diagonalizes C 
		 * @param C the set of matrices to be diagonalized
		 * @param V0 an estimate of the matrix V
		 * @param eps machine epsilon or desired epsilon
		 * @param itermax maximum number of iterations
		 * @return V the matrix the best diagonalizes C 
		 */
		virtual SGMatrix<float64_t> compute(SGNDArray<float64_t> &C,
						   SGMatrix<float64_t> V0= SGMatrix<float64_t>(NULL,0,0),
						   double eps=CMath::MACHINE_EPSILON,
						   int itermax=200)
		{
			m_V = diagonalize(C,V0,eps,itermax);
			return m_V;	
		}

		/** @return object name */
		virtual const char* get_name() const { return "UWedge"; }
};
}
#endif //HAVE_EIGEN3
#endif //UWEDGE_H_ 
