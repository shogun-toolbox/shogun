/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser, Sergey Lisitsyn
 */

#ifndef QDIAG_H_
#define QDIAG_H_

#include <shogun/lib/config.h>

#include <shogun/mathematics/ajd/ApproxJointDiagonalizer.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>

namespace shogun
{

/** @brief Class QDiag
 *
 * An Approximate Joint Diagonalizer (AJD) Implementation
 *
 * Vollgraf, R., & Obermayer, K. (2006).
 * Quadratic optimization for simultaneous matrix diagonalization.
 * Signal Processing, IEEE Transactions on, 54(9), 3270-3278.
 *
 */
class QDiag : public RandomMixin<ApproxJointDiagonalizer>
{
	public:

		/** constructor */
		QDiag()
		{
		}

		/** destructor */
		virtual ~QDiag()
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
							SGMatrix<float64_t> V0,
							double eps=Math::MACHINE_EPSILON,
							int itermax=200);

		template <typename PRNG>
		static SGMatrix<float64_t> diagonalize(SGNDArray<float64_t> C,
							PRNG& prng,
							double eps=Math::MACHINE_EPSILON,
							int itermax=200)
		{
			int N = C.dims[0];
			auto V = SGMatrix<float64_t>(N,N);

			random::fill_array(V, NormalDistribution<float64_t>(), prng);
			return diagonalize_impl(C, V, eps, itermax);
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
						   int itermax=200)
		{
			int N = C.dims[0];
			if(V0.num_rows != N || V0.num_cols != N)
				m_V = diagonalize(C,m_prng,eps,itermax);
			else
				m_V = diagonalize(C, V0, eps, itermax);
			return m_V;
		}

		/** @return object name */
		virtual const char* get_name() const { return "QDiag"; }
	
	private:
		static SGMatrix<float64_t> diagonalize_impl(SGNDArray<float64_t>& C,
					SGMatrix<float64_t>& V,
					double eps,
					int itermax);
};
}
#endif //QDIAG_H_
