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

class Kernel;

/** TODO doc */
class KernelStructuredOutputMachine : public StructuredOutputMachine
{
	public:
		/** default constructor  */
		KernelStructuredOutputMachine();

		/** standard constructor
		 *
		 * @param model structured model with application specific functions
		 * @param labs structured labels
		 * @param kernel kernel
		 */
		KernelStructuredOutputMachine(std::shared_ptr<StructuredModel> model, std::shared_ptr<StructuredLabels> labs, std::shared_ptr<Kernel> kernel);

		/** destructor */
		virtual ~KernelStructuredOutputMachine();

		/** set kernel
		 *
		 * @param f kernel
		 */
		void set_kernel(std::shared_ptr<Kernel> f);

		/** get kernel
		 *
		 * @return kernel
		 */
		std::shared_ptr<Kernel> get_kernel() const;

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
		std::shared_ptr<Kernel> m_kernel;

}; /* class KernelStructuredOutputMachine */

} /* namespace shogun */

#endif /* _KERNEL_STRUCTURED_OUTPUT_MACHINE__H__ */
