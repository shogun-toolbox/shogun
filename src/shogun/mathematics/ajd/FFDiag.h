/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Fernando Iglesias, Bjoern Esser, 
 *          Sergey Lisitsyn, Soeren Sonnenburg
 */

#ifndef FFDIAG_H_
#define FFDIAG_H_

#include <shogun/lib/config.h>


#include <shogun/mathematics/ajd/ApproxJointDiagonalizer.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

/** @brief Class FFDiag
 *
 * An Approximate Joint Diagonalizer (AJD) Implementation
 *
 * Ziehe, A., Laskov, P., Nolte, G., & Mueller, K. R. (2004).
 * A fast algorithm for joint diagonalization with non-orthogonal transformations
 * and its application to blind source separation.
 * The Journal of Machine Learning Research, 5, 777-800.
 *
 */
class FFDiag : public ApproxJointDiagonalizer
{
	public:

		/** constructor */
		FFDiag()
		{
		}

		/** destructor */
		virtual ~FFDiag()
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
		virtual const char* get_name() const { return "FFDiag"; }
};
}
#endif //FFDIAG_H_
