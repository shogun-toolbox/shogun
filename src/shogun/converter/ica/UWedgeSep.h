/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser
 */

#ifndef UWEDGESEP_H_
#define UWEDGESEP_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGNDArray.h>
#include <shogun/features/Features.h>
#include <shogun/converter/ica/ICAConverter.h>

namespace shogun
{

class CFeatures;

/** @brief class UWedgeSep
 *
 * Implements the UWedge algorithm for Independent
 * Component Analysis (ICA) and Blind Source
 * Separation (BSS).
 *
 * Tichavsky, P., & Yeredor, A. (2009).
 * Fast approximate joint diagonalization incorporating weight matrices.
 * Signal Processing, IEEE Transactions on, 57(3), 878-891.
 *
 */
class CUWedgeSep: public CICAConverter
{
	public:

		/** constructor */
		CUWedgeSep();

		/** destructor */
		virtual ~CUWedgeSep();

		virtual void fit(CFeatures* features);

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
		virtual const char* get_name() const { return "UWedgeSep"; };

	protected:

		/** init */
		void init();

	private:

		/** tau vector */
		SGVector<float64_t> m_tau;

		/** cov matrices */
		SGNDArray<float64_t> m_covs;
};
}
#endif // UWEDGESEP
