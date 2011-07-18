/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KERNELPERCEPTRON_H___
#define _KERNELPERCEPTRON_H___

#include <stdio.h>
#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/machine/KernelMachine.h>

namespace shogun
{
/** @brief Class KernelPerceptron -
 * currently unfinished implementation of a Kernel Perceptron
 */
class CKernelPerceptron : public CKernelMachine
{
	public:
		/** constructor */
		CKernelPerceptron();
		virtual ~CKernelPerceptron();

		/** train kernel perceptron classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** classify one specific example
		 *
		 * @param num which example to classify
		 * @return classified value
		 */
		virtual float64_t classify_example(int32_t num);

		/** load KernelPerceptron from file
		 *
		 * @param srcfile file to load from
		 * @return if load was successful
		 */
		virtual bool load(FILE* srcfile);

		/** save KernelPerceptron to file
		 *
		 * @param dstfile file to save to
		 * @return if save was successful
		 */
		virtual bool save(FILE* dstfile);

		/** get classifier type
		 *
		 * @return classifier type KERNELPERCEPTRON
		 */
		inline virtual EClassifierType get_classifier_type()
		{
			return CT_KERNELPERCEPTRON;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "KernelPerceptron"; }
};
}
#endif
