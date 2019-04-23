/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser
 */

#ifndef FFSEP_H_
#define FFSEP_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGNDArray.h>
#include <shogun/features/Features.h>
#include <shogun/converter/ica/ICAConverter.h>

namespace shogun
{

class Features;

/** @brief class FFSep
 *
 * Implements the FFSep algorithm for Independent
 * Component Analysis (ICA) and Blind Source
 * Separation (BSS).
 *
 * Ziehe, A., Laskov, P., Nolte, G., & Müller, K. R. (2004).
 * A fast algorithm for joint diagonalization with non-orthogonal transformations
 * and its application to blind source separation.
 * The Journal of Machine Learning Research, 5, 777-800.
 *
 */
class FFSep: public ICAConverter
{
	public:

		/** constructor */
		FFSep();

		/** destructor */
		virtual ~FFSep();


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
		virtual const char* get_name() const { return "FFSep"; };

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
#endif // FFSEP
