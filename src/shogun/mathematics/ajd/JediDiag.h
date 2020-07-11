/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser
 */

#ifndef JEDIDIAG_H_
#define JEDIDIAG_H_

#include <shogun/lib/config.h>


#include <shogun/mathematics/ajd/ApproxJointDiagonalizer.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

/** @brief Class Jedi
 *
 * An Approximate Joint Diagonalizer (AJD) Implementation
 *
 * Souloumiac, A. (2009).
 * Nonorthogonal joint diagonalization by combining givens and hyperbolic rotations.
 * Signal Processing, IEEE Transactions on, 57(6), 2222-2231.
 *
 */
class JediDiag : public ApproxJointDiagonalizer
{
	public:

		/** constructor */
		JediDiag()
		{
		}

		/** destructor */
		~JediDiag() override
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
		SGMatrix<float64_t> compute(SGNDArray<float64_t> C,
						   SGMatrix<float64_t> V0 = SGMatrix<float64_t>(NULL,0,0,false),
						   double eps=Math::MACHINE_EPSILON,
						   int itermax=200) override
		{
			m_V = diagonalize(C,V0,eps,itermax);
			return m_V;
		}

		/** @return object name */
		const char* get_name() const override { return "JediDiag"; }
};
}
#endif //JEDIDIAG_H_
