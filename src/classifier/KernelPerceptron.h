/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KERNELPERCEPTRON_H___
#define _KERNELPERCEPTRON_H___

#include <stdio.h>
#include "lib/common.h"
#include "features/Features.h"
#include "kernel/KernelMachine.h"

/** class KernelPerceptron */
class CKernelPerceptron : public CKernelMachine
{
	public:
		/** constructor */
		CKernelPerceptron();
		virtual ~CKernelPerceptron();

		/** train KernelPerceptron */
		virtual bool train();

		/** classify one specific example
		 *
		 * @param num which example to classify
		 * @return classified value
		 */
		virtual DREAL classify_example(int32_t num);

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
};
#endif

