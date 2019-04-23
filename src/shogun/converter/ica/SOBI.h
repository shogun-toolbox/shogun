/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser
 */

#ifndef SOBI_H_
#define SOBI_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGNDArray.h>
#include <shogun/features/Features.h>
#include <shogun/converter/ica/ICAConverter.h>

namespace shogun
{

class Features;

/** @brief class SOBI
 *
 * Implements the Second Order Blind Identification (SOBI)
 * algorithm for Independent Component Analysis (ICA) and
 * Blind Source Separation (BSS). This algorithm is also
 * sometime refered to as Temporal Decorrelation Separation
 * (TDSep).
 *
 * Belouchrani, A., Abed-Meraim, K., Cardoso, J. F., & Moulines, E. (1997).
 * A blind source separation technique using second-order statistics.
 * Signal Processing, IEEE Transactions on, 45(2), 434-444.
 *
 */
class SOBI: public ICAConverter
{
	public:

		/** constructor */
		SOBI();

		/** destructor */
		virtual ~SOBI();

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
		virtual const char* get_name() const { return "SOBI"; };

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
#endif // SOBI
