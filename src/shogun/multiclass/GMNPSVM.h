/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Yuyu Zhang, Bjoern Esser
 */

#ifndef _GMNPSVM_H___
#define _GMNPSVM_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/multiclass/MulticlassSVM.h>
#include <shogun/features/Features.h>

namespace shogun
{
/** @brief Class GMNPSVM implements a one vs. rest MultiClass SVM.
 *
 * It uses CGMNPLib for training (in true multiclass-SVM fashion).
 */
class SHOGUN_EXPORT CGMNPSVM : public CMulticlassSVM
{
	void init();

	public:
		/** default constructor */
		CGMNPSVM();

		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CGMNPSVM(float64_t C, CKernel* k, CLabels* lab);

		/** default destructor */
		virtual ~CGMNPSVM();

		/** get classifier type
		 *
		 * @return classifier type GMNPSVM
		 */
		virtual EMachineType get_classifier_type() { return CT_GMNPSVM; }

		/** required for CMKLMulticlass constraint computation
		 *
		 *  @param y height of basealphas
		 *  @param x width of basealphas
		 *
		 *  @return basealphas basealphas[k][j] is the alpha for class
		 *	        k and sample j which is untransformed compared to
		 *	        the alphas stored in CSVM* members
		 */
		float64_t* get_basealphas_ptr(index_t* y, index_t* x);

		/** @return object name */
		virtual const char* get_name() const { return "GMNPSVM"; }

	protected:
		/** train SVM
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	protected:
		/** required for CMKLMulticlass
		 * stores the untransformed alphas of this algorithm
		 * whereas CSVM* members stores a transformed version of it
		 * m_basealphas[k][j] is the alpha for class k and sample j
		 */
		// is the basic untransformed alpha, needed for MKL
		float64_t* m_basealphas;
		/** base alphas y */
		index_t m_basealphas_y;
		/** base alphas x */
		index_t m_basealphas_x;
};
}
#endif
