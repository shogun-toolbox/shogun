/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2009 Soeren Sonnnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _AUCKERNEL_H___
#define _AUCKERNEL_H___

#include <lib/common.h>
#include <kernel/DotKernel.h>
#include <features/DenseFeatures.h>
#include <labels/Labels.h>

namespace shogun
{
	class CLabels;
	template <class T> class CDenseFeatures;

/** @brief The AUC kernel can be used to maximize the area under the receiver operator
 * characteristic curve (AUC) instead of margin in SVM training.
 *
 * It takes as argument a sub-kernel and Labels based on which number of
 * positive labels times number of negative labels many ``virtual'' examples
 * are created that ensure that all positive examples get a higher score than
 * all negative examples in training.
 */
class CAUCKernel: public CDotKernel
{
	void init();

	public:
		/** default constructor  */
		CAUCKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param subkernel the subkernel
		 */
		CAUCKernel(int32_t size, CKernel* subkernel);

		/** destructor */
		virtual ~CAUCKernel();

		/** initialize kernel based on current labeling and subkernel
		 *
		 * @param labels - current labeling
		 * @return new label object to be used together with this kernel in SVM
		 * training for AUC maximization
		 */
		CLabels* setup_auc_maximization(CLabels* labels);

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** return what type of kernel we are
		 *
		 * @return kernel type AUC
		 */
		virtual EKernelType get_kernel_type() { return K_AUC; }

		/** return the kernel's name
		 *
		 * @return name AUC
		 */
		virtual const char* get_name() const { return "AUCKernel" ; }

		/** return feature class the kernel can deal with
		 *
		 * @return feature class SIMPLE
		 */
		virtual EFeatureClass get_feature_class() { return C_DENSE; }

		/** return feature type the kernel can deal with
		 *
		 * @return word feature type
		 */
		virtual EFeatureType get_feature_type() { return F_WORD; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

	protected:
		/** the subkernel */
		CKernel* subkernel;
};
}
#endif /* _AUCKERNEL_H__ */
