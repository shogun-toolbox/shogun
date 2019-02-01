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

class CFeatures;
class CKernel;

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
		CKernelMulticlassMachine(CMulticlassStrategy *strategy, CKernel* kernel, CMachine* machine, CLabels* labs);

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
