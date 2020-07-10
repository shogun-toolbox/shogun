/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _NATIVEMULTICLASSMACHINE_H___
#define _NATIVEMULTICLASSMACHINE_H___

#include <shogun/lib/config.h>

#include <shogun/machine/MulticlassMachine.h>

namespace shogun
{

/** @brief experimental abstract native multiclass machine class */
class NativeMulticlassMachine : public MulticlassMachine
{
	public:
		/** default constructor  */
		NativeMulticlassMachine()
		{
		}

		/** destructor */
		~NativeMulticlassMachine() override
		{
		}

		/** get name */
		const char* get_name() const override
		{
			return "NativeMulticlassMachine";
		}

	protected:
		/** init strategy */
		void init_strategy() { }

		/** clear machines */
		void clear_machines() { }

		/** abstract init machine for training method */
		bool init_machine_for_train(std::shared_ptr<Features> data) override { return true; }

		/** abstract init machines for applying method */
		bool init_machines_for_apply(std::shared_ptr<Features> data) override { return true; }

		/** check whether machine is ready */
		bool is_ready() override { return true; }

		/** obtain machine from trained one */
		std::shared_ptr<Machine> get_machine_from_trained(std::shared_ptr<Machine> machine) const override { return NULL; }

		/** get num rhs vectors */
		int32_t get_num_rhs_vectors() const override { return 0; }

		/** set subset to the features of the machine, deletes old one
		 *
		 * @param subset subset indices to set
		 */
		void add_machine_subset(SGVector<index_t> subset) override { }

		/** deletes any subset set to the features of the machine */
		void remove_machine_subset() override { }

		/** whether the machine is acceptable in set_machine */
		bool is_acceptable_machine(std::shared_ptr<Machine >machine) override { return true; }

};
}
#endif
