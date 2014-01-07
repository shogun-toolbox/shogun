/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2008 Center for Machine Perception, CTU FEL Prague
 */

#ifndef _GMNPSVM_H___
#define _GMNPSVM_H___

#include <lib/common.h>
#include <multiclass/MulticlassSVM.h>
#include <features/Features.h>

namespace shogun
{
/** @brief Class GMNPSVM implements a one vs. rest MultiClass SVM.
 *
 * It uses CGMNPLib for training (in true multiclass-SVM fashion).
 */
class CGMNPSVM : public CMulticlassSVM
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
