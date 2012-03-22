/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2012 Soeren Sonnenburg and Sergey Lisitsyn
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
		CKernelMulticlassMachine() : CMulticlassMachine(), m_kernel(NULL)
		{
			m_parameters->add((CSGObject**)&m_kernel,"m_kernel");
		}

		/** standard constructor
		 * @param strategy multiclass strategy
		 * @param kernel kernel
		 * @param machine kernel machine
		 * @param labs labels
		 */
		CKernelMulticlassMachine(EMulticlassStrategy strategy, CKernel* kernel, CKernelMachine* machine, CLabels* labs) :
			CMulticlassMachine(strategy,(CMachine*)machine,labs), m_kernel(NULL)
		{
			set_kernel(kernel);
			m_parameters->add((CSGObject**)&m_kernel,"m_kernel");
		}

		/** destructor */
		virtual ~CKernelMulticlassMachine()
		{
			SG_UNREF(m_kernel);
		}

		/** get name */
		virtual const char* get_name() const 
		{
			return "KernelMulticlassMachine";
		}

		/** set kernel
		 *
		 * @param k kernel
		 */
		void set_kernel(CKernel* k)
		{
			SG_REF(k);
			SG_UNREF(m_kernel);
			m_kernel=k;
		}

		/** get kernel
		 *
		 * @return kernel
		 */
		CKernel* get_kernel()
		{
			SG_REF(m_kernel);
			return m_kernel;
		}

	protected:

		/** init machine for training with kernel init */
		virtual bool init_machine_for_train(CFeatures* data)
		{
			if (data)
				m_kernel->init(data,data);

			((CKernelMachine*)m_machine)->set_kernel(m_kernel);

			return true;
		}

		/** init machines for applying with kernel init */
		virtual bool init_machines_for_apply(CFeatures* data)
		{
			if (data)
				m_kernel->init(m_kernel->get_lhs(),data);

			for (int32_t i=0; i<m_machines.vlen; i++)
				((CKernelMachine*)m_machines[i])->set_kernel(m_kernel);
			return true;
		}

		/** check kernel availability */
		virtual bool is_ready()
		{
			if (m_kernel && m_kernel->get_num_vec_lhs() && m_kernel->get_num_vec_rhs())
					return true;

			return false;
		}

		/** construct kernel machine from given kernel machine */
		virtual CMachine* get_machine_from_trained(CMachine* machine)
		{
			return new CKernelMachine((CKernelMachine*)machine);
		}

		/** return number of rhs feature vectors */
		virtual int32_t get_num_rhs_vectors()
		{
			return m_kernel->get_num_vec_rhs();
		}

		/** set subset to the features of the machine, deletes old one
		 *
		 * @param subset subset instance to set
		 */
		virtual void set_machine_subset(CSubset* subset)
		{
			SG_NOTIMPLEMENTED;
		}

		/** deletes any subset set to the features of the machine */
		virtual void remove_machine_subset()
		{
			SG_NOTIMPLEMENTED;
		}

	protected:

		/** kernel */
		CKernel* m_kernel;

};
}
#endif
