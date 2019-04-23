/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Bjoern Esser
 */

#ifndef FASTICA_H_
#define FASTICA_H_

#include <shogun/lib/config.h>
#include <shogun/converter/ica/ICAConverter.h>
#include <shogun/features/Features.h>
#include <shogun/mathematics/RandomMixin.h>

namespace shogun
{

class Features;

/** @brief class FastICA
 *
 * Implements the FastICA (Independent
 * Component Analysis) algorithm
 *
 * A. Hyvarinen and E. Oja, Independent Component Analysis:
 * Algorithms and Applications, Neural Networks, 13(4-5), 2000,
 * pp. 411-430`
 */
class FastICA: public RandomMixin<ICAConverter>
{
	public:

		/** constructor */
		FastICA();

		/** destructor */
		virtual ~FastICA();

		/** setter for whiten flag
		 * whether to whiten the data or not
		 * @param whiten
		 */
		void set_whiten(bool whiten);

		/** getter for whiten flag
		 * @return whiten
		 */
		bool get_whiten() const;

		/** @return object name */
		virtual const char* get_name() const { return "FastICA"; };

	protected:

		/** init */
		void init();

		virtual void fit_dense(std::shared_ptr<DenseFeatures<float64_t>> features);

	private:

		/** whiten */
		bool whiten;

};
}
#endif // FASTICA
