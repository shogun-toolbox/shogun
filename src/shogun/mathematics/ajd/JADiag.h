/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser, Sergey Lisitsyn
 */

#ifndef JADIAG_H_
#define JADIAG_H_

#include <shogun/lib/config.h>


#include <shogun/mathematics/ajd/ApproxJointDiagonalizer.h>
#include <shogun/mathematics/Math.h>

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
class JADiag : public ApproxJointDiagonalizer
{
	public:

		/** constructor */
		JADiag()
		{
		}

		/** destructor */
		virtual ~JADiag()
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
							double eps=Math::MACHINE_EPSILON,
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
						   double eps=Math::MACHINE_EPSILON,
						   int itermax=200)
		{
			m_V = diagonalize(C,V0,eps,itermax);
			return m_V;
		}

		/** @return object name */
		virtual const char* get_name() const { return "JADiag"; }
};
}
#endif //JADIAG_H_
