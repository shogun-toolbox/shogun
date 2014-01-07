/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _KERNEL_STRUCTURED_OUTPUT_MACHINE__H__
#define _KERNEL_STRUCTURED_OUTPUT_MACHINE__H__

#include <machine/StructuredOutputMachine.h>
#include <kernel/Kernel.h>

namespace shogun
{

/** TODO doc */
class CKernelStructuredOutputMachine : public CStructuredOutputMachine
{
	public:
		/** default constructor  */
		CKernelStructuredOutputMachine();

		/** standard constructor
		 *
		 * @param model structured model with application specific functions
		 * @param labs structured labels
		 * @param kernel kernel
		 */
		CKernelStructuredOutputMachine(CStructuredModel* model, CStructuredLabels* labs, CKernel* kernel);

		/** destructor */
		virtual ~CKernelStructuredOutputMachine();

		/** set kernel
		 *
		 * @param f kernel
		 */
		void set_kernel(CKernel* f);

		/** get kernel
		 *
		 * @return kernel
		 */
		CKernel* get_kernel() const;

		/** @return object name */
		virtual const char* get_name() const
		{
			return "KernelStructuredOutputMachine";
		}

	private:
		/** register class members */
		void register_parameters();

	protected:
		/** kernel */
		CKernel* m_kernel;

}; /* class CKernelStructuredOutputMachine */

} /* namespace shogun */

#endif /* _KERNEL_STRUCTURED_OUTPUT_MACHINE__H__ */
