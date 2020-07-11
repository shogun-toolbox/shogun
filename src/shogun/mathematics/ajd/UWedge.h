/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Sergey Lisitsyn, Bjoern Esser
 */

#ifndef UWEDGE_H_
#define UWEDGE_H_

#include <shogun/lib/config.h>


#include <shogun/mathematics/ajd/ApproxJointDiagonalizer.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{

/** @brief Class UWedge
 *
 * An Approximate Joint Diagonalizer (AJD) Implementation
 *
 * Tichavsky, P., & Yeredor, A. (2009).
 * Fast approximate joint diagonalization incorporating weight matrices.
 * Signal Processing, IEEE Transactions on, 57(3), 878-891.
 *
 */
class UWedge : public ApproxJointDiagonalizer
{
	public:

		/** constructor */
		UWedge()
		{
		}

		/** destructor */
		~UWedge() override
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
								double eps=1e-12,
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
						   double eps=1e-12,
						   int itermax=200) override
		{
			m_V = diagonalize(C,V0,eps,itermax);
			return m_V;
		}

		/** @return object name */
		const char* get_name() const override { return "UWedge"; }
};
}
#endif //UWEDGE_H_
