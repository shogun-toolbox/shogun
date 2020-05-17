/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _AUCKERNEL_H___
#define _AUCKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/common.h>

namespace shogun
{
	class Labels;
	template <class T>
	class DenseFeatures;

	/** @brief The AUC kernel can be used to maximize the area under the
	 * receiver operator characteristic curve (AUC) instead of margin in SVM
	 * training.
	 *
	 * It takes as argument a sub-kernel and Labels based on which number of
	 * positive labels times number of negative labels many ``virtual'' examples
	 * are created that ensure that all positive examples get a higher score
	 * than all negative examples in training.
	 */
	class AUCKernel : public DotKernel
	{
	public:
	
		AUCKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param subkernel the subkernel
		 * @param labels the labels for AUC maximization
		 */
		AUCKernel(int32_t size, std::shared_ptr<Kernel> subkernel, std::shared_ptr<Labels> labels);

		virtual ~AUCKernel();

		/** initialize kernel based on current labeling and subkernel
		 *
		 * @param labels - current labeling
		 * @return whether new labels with AUC maximisation have been learnt
		 */
		bool setup_auc_maximization();

		virtual EKernelType get_kernel_type() { return K_AUC; }
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		virtual const char* get_name() const { return "AUCKernel"; }

		virtual EFeatureClass get_feature_class() { return C_DENSE; }
		
		virtual EFeatureType get_feature_type() { return F_WORD; }

	protected:
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

	protected:
		std::shared_ptr<Kernel> m_subkernel;
		std::shared_ptr<Labels> m_labels;
	};
} // namespace shogun
#endif /* _AUCKERNEL_H__ */
