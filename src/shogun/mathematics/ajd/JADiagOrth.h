/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Sergey Lisitsyn, Bjoern Esser, 
 *          Viktor Gal
 */

#ifndef JADIAGORTH_H_
#define JADIAGORTH_H_

#include <shogun/lib/config.h>


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
class JADiagOrth : public ApproxJointDiagonalizer
{
	public:

		/** constructor */
		JADiagOrth()
		{
		}

		/** destructor */
		virtual ~JADiagOrth()
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
		virtual const char* get_name() const { return "JADiagOrth"; }
};
}
#endif //JADIAGORTH_H_
