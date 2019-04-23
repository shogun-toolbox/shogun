/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser, Sergey Lisitsyn
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
class ApproxJointDiagonalizer : public SGObject
{
	public:

		/** constructor */
		ApproxJointDiagonalizer() : SGObject()
		{
		};

		/** destructor */
		virtual ~ApproxJointDiagonalizer()
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
						   double eps=Math::MACHINE_EPSILON,
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
