/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn, Viktor Gal
 */

#ifndef _SVMLIGHTONECLASS_H___
#define _SVMLIGHTONECLASS_H___

#include <shogun/lib/config.h>
#include <shogun/machine/Machine.h>

#ifdef USE_SVMLIGHT
#include <shogun/classifier/svm/SVMLight.h>
#endif //USE_SVMLIGHT

#ifdef USE_SVMLIGHT
namespace shogun
{
/** @brief Trains a one class C SVM
 *
 * \sa SVMLight
 */
class SVMLightOneClass: public SVMLight
{
	public:
		/** default constructor */
		SVMLightOneClass();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 */
		SVMLightOneClass(float64_t C, std::shared_ptr<Kernel> k);

		/** default destructor */
		virtual ~SVMLightOneClass() { }

		/** get classifier type
		 *
		 * @return classifier type LIGHTONECLASS
		 */
		virtual EMachineType get_classifier_type() { return CT_LIGHTONECLASS; }

		/** Returns the name of the SGSerializable instance.
		 *
		 * @return name of the SGSerializable
		 */
		virtual const char* get_name() const { return "SVMLightOneClass"; }

	protected:
		/** train one class svm
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based regressors are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(std::shared_ptr<Features> data=NULL);
};
}
#endif //USE_SVMLIGHT
#endif // _SVMLIGHTONECLASS_H___
