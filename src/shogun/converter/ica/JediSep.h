/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser
 */

#ifndef JEDISEP_H_
#define JEDISEP_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGNDArray.h>
#include <shogun/features/Features.h>
#include <shogun/converter/ica/ICAConverter.h>

namespace shogun
{

class Features;

/** @brief class JediSep
 *
 * Implements the JediSep algorithm for Independent
 * Component Analysis (ICA) and Blind Source
 * Separation (BSS).
 *
 * Souloumiac, A. (2009).
 * Nonorthogonal joint diagonalization by combining givens and hyperbolic rotations.
 * Signal Processing, IEEE Transactions on, 57(6), 2222-2231.
 *
 */
class JediSep: public ICAConverter
{
	public:

		/** constructor */
		JediSep();

		/** destructor */
		virtual ~JediSep();

		/** getter for tau parameter
		 * @return tau vector
		 */
		SGVector<float64_t> get_tau() const;

		/** setter for tau parameter
		 * @param tau vector
		 */
		void set_tau(SGVector<float64_t> tau);

		/** getter for time sep cov matrices
		 * @return cov matrices
		 */
		SGNDArray<float64_t> get_covs() const;

		/** @return object name */
		virtual const char* get_name() const { return "JediSep"; };

	protected:

		/** init */
		void init();

		virtual void fit_dense(std::shared_ptr<DenseFeatures<float64_t>> features);

	private:

		/** tau vector */
		SGVector<float64_t> m_tau;

		/** cov matrices */
		SGNDArray<float64_t> m_covs;
};
}
#endif // JEDISEP
