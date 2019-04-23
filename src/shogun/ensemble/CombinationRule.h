/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Bjoern Esser, Viktor Gal
 */

#ifndef _COMBINATION_RULE_H_
#define _COMBINATION_RULE_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>

namespace shogun
{
	/**
	 * @brief CombinationRule abstract class
	 * The CombinationRule defines an interface to how to combine the
	 * classification or regression outputs of an ensemble of Machines.
	 */
	class CombinationRule : public SGObject
	{
		public:
			/** default ctor */
			CombinationRule();

			virtual ~CombinationRule();

			/**
			 * Combines a matrix of an ensemble of Machines output, where each
			 * column is a given Machine's classification/regression output
			 * for the input Features.
			 *
			 * @param ensemble_result SGMatrix
			 * @return a vector where the nth element is the combined value of the Machines for the nth feature vector
			 */
			virtual SGVector<float64_t> combine(const SGMatrix<float64_t>& ensemble_result) const = 0;

			/**
			 * Combines a vector of Machine ouputs for a given feature vector.
			 * The nth element of the vector is the nth Machine's output in the ensemble.
			 *
			 * @param ensemble_result SGVector<float64_t> with the Machine's output
			 * @return the combined value
			 */
			virtual float64_t combine(const SGVector<float64_t>& ensemble_result) const = 0;

			/** name **/
			virtual const char* get_name() const { return "CombinationRule"; }
	};
}

#endif /* _COMBINATION_RULE_H_ */
