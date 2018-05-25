/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser
 */

#ifndef JADE_H_
#define JADE_H_

#include <shogun/lib/config.h>
#include <shogun/converter/ica/ICAConverter.h>
#include <shogun/features/Features.h>

namespace shogun
{

class CFeatures;

//#define DEBUG_JADE

/** @brief class Jade
 *
 * Implements the JADE algorithm for Independent
 * Component Analysis (ICA) and Blind Source
 * Separation (BSS).
 *
 * Cardoso, J. F., & Souloumiac, A. (1993).
 * Blind beamforming for non-Gaussian signals.
 * In IEE Proceedings F (Radar and Signal Processing)
 * (Vol. 140, No. 6, pp. 362-370). IET Digital Library.
 *
 */
class CJade: public CICAConverter
{
	public:

		/** constructor */
		CJade();

		/** destructor */
		virtual ~CJade();

		virtual void fit(CFeatures* features);

		/** getter for cumulant_matrix
		 * @return cumulant_matrix
		 */
		SGMatrix<float64_t> get_cumulant_matrix() const;

		/** @return object name */
		virtual const char* get_name() const { return "Jade"; };

	protected:

		/** init */
		void init();

	private:

		/** cumulant_matrix */
		SGMatrix<float64_t> m_cumulant_matrix;
};
}
#endif // JADE
