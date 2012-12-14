/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2012 Soeren Sonnenburg and Sergey Lisitsyn
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 1999-2012 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KERNELMULTICLASSMACHINE_H___
#define _KERNELMULTICLASSMACHINE_H___

#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/machine/KernelMachine.h>
#include <shogun/machine/MulticlassMachine.h>

namespace shogun
{

class CKernel;
class CKernelMachine;

/** @brief generic kernel multiclass */
class CKernelMulticlassMachine : public CMulticlassMachine
{
	public:
		/** default constructor  */
		CKernelMulticlassMachine();

		/** standard constructor
		 * @param strategy multiclass strategy
		 * @param kernel kernel
		 * @param machine kernel machine
		 * @param labs labels
		 */
		CKernelMulticlassMachine(CMulticlassStrategy *strategy, CKernel* kernel, CKernelMachine* machine, CLabels* labs);

		/** destructor */
		virtual ~CKernelMulticlassMachine();

		/** get name */
		virtual const char* get_name() const
		{
			return "KernelMulticlassMachine";
		}

		/** set kernel
		 *
		 * @param k kernel
		 */
		void set_kernel(CKernel* k);

		/** get kernel
		 *
		 * @return kernel
		 */
		CKernel* get_kernel();

		/** Stores feature data of underlying model.
		 *
		 * Need to store the SVs for all sub-machines. We make a union of the
		 * SVs for all sub-machines, store the union and adjust the
		 * sub-machines to index into the union.
		 */
		virtual void store_model_features();

	protected:

		/** init machine for training with kernel init */
		virtual bool init_machine_for_train(CFeatures* data);

		/** init machines for applying with kernel init */
		virtual bool init_machines_for_apply(CFeatures* data);

		/** check kernel availability */
		virtual bool is_ready();

		/** construct kernel machine from given kernel machine */
		virtual CMachine* get_machine_from_trained(CMachine* machine);

		/** return number of rhs feature vectors */
		virtual int32_t get_num_rhs_vectors();

		/** set subset to the features of the machine, deletes old one
		 *
		 * @param subset subset indices to set
		 */
		virtual void add_machine_subset(SGVector<index_t> subset);

		/** deletes any subset set to the features of the machine */
		virtual void remove_machine_subset();

	protected:

		/** kernel */
		CKernel* m_kernel;

};
}
#endif
