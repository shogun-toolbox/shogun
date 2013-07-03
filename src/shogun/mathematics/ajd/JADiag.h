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

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGNDArray.h>

#include <limits>

namespace shogun
{

/** @brief Class JADiag
 *
 * An Approximate Joint Diagonalizer (AJD) Implementation
 */
class CJADiag : public CApproxJointDiagonalizer
{
	public:
	
		/** constructor */
		CJADiag()
		{
		};

		/** destructor */
		virtual ~CJADiag()
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
       							SGMatrix<float64_t> *V0=NULL,
							double eps=std::numeric_limits<double>::epsilon(),
							int itermax=200);

		/** Computes the matrix V that best diagonalizes C 
		 * @param C the set of matrices to be diagonalized
		 * @param V0 an estimate of the matrix V
		 * @param eps machine epsilon or desired epsilon
		 * @param itermax maximum number of iterations
		 * @return V the matrix the best diagonalizes C 
		 */
		virtual SGMatrix<float64_t> compute(SGNDArray<float64_t> &C,
						   SGMatrix<float64_t> *V0=NULL,
						   double eps=std::numeric_limits<double>::epsilon(),
						   int itermax=200)
		{
			m_V = diagonalize(C,V0,eps,itermax);
			return m_V;	
		}
};
}
#endif //JADIAG_H_ 
