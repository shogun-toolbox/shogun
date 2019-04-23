/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn
 */

#ifndef _CPLEXSVM_H___
#define _CPLEXSVM_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>

#ifdef USE_CPLEX
#include <shogun/classifier/svm/SVM.h>
#include <shogun/lib/Cache.h>

namespace shogun
{
#define IGNORE_IN_CLASSLIST
/** @brief CplexSVM a SVM solver implementation based on cplex (unfinished). */
IGNORE_IN_CLASSLIST class CCPLEXSVM : public SVM
{
	public:
		CCPLEXSVM();
		virtual ~CCPLEXSVM();

		virtual EMachineType get_classifier_type() { return CT_CPLEXSVM; }

	protected:
		/** train SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(Features* data=NULL);
};
}
#endif
#endif
