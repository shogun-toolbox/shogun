/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Yuyu Zhang, Shell Hu, Thoralf Klein, 
 *          Bjoern Esser, Soeren Sonnenburg
 */

#ifndef _KERNEL_STRUCTURED_OUTPUT_MACHINE__H__
#define _KERNEL_STRUCTURED_OUTPUT_MACHINE__H__

#include <shogun/lib/config.h>

#include <shogun/machine/StructuredOutputMachine.h>

namespace shogun
{

class CKernel;

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
