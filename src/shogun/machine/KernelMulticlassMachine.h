/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Chiyuan Zhang, Heiko Strathmann, Fernando Iglesias,
 *          Soeren Sonnenburg, Evan Shelhamer, Yuyu Zhang, Thoralf Klein
 */

#ifndef _KERNELMULTICLASSMACHINE_H___
#define _KERNELMULTICLASSMACHINE_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/MulticlassMachine.h>

namespace shogun
{

class Features;
class Kernel;

/** @brief generic kernel multiclass */
class KernelMulticlassMachine : public MulticlassMachine
{
	public:
		/** default constructor  */
		KernelMulticlassMachine();

		/** standard constructor
		 * @param strategy multiclass strategy
		 * @param kernel kernel
		 * @param machine kernel machine
		 * @param labs labels
		 */
		KernelMulticlassMachine(std::shared_ptr<MulticlassStrategy >strategy, std::shared_ptr<Kernel> kernel, std::shared_ptr<Machine> machine, std::shared_ptr<Labels> labs);

		/** destructor */
		virtual ~KernelMulticlassMachine();

		/** get name */
		virtual const char* get_name() const
		{
			return "KernelMulticlassMachine";
		}

		/** set kernel
		 *
		 * @param k kernel
		 */
		void set_kernel(const std::shared_ptr<Kernel>& k);

		/** get kernel
		 *
		 * @return kernel
		 */
		std::shared_ptr<Kernel> get_kernel() const;

		/** Stores feature data of underlying model.
		 *
		 * Need to store the SVs for all sub-machines. We make a union of the
		 * SVs for all sub-machines, store the union and adjust the
		 * sub-machines to index into the union.
		 */
		virtual void store_model_features();

	protected:

		/** init machine for training with kernel init */
		virtual bool init_machine_for_train(std::shared_ptr<Features> data);

		/** init machines for applying with kernel init */
		virtual bool init_machines_for_apply(std::shared_ptr<Features> data);

		/** check kernel availability */
		virtual bool is_ready();

		/** construct kernel machine from given kernel machine */
		virtual std::shared_ptr<Machine> get_machine_from_trained(std::shared_ptr<Machine> machine) const;

		/** return number of rhs feature vectors */
		virtual int32_t get_num_rhs_vectors() const;

		/** set subset to the features of the machine, deletes old one
		 *
		 * @param subset subset indices to set
		 */
		virtual void add_machine_subset(SGVector<index_t> subset);

		/** deletes any subset set to the features of the machine */
		virtual void remove_machine_subset();

	protected:

		/** kernel */
		std::shared_ptr<Kernel> m_kernel;

};
}
#endif
