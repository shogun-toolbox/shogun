/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2011 Soeren Sonnenburg
 * Written (W) 2012 Fernando José Iglesias García and Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn, Fernando José Iglesias Garcia
 */

#ifndef _NATIVEMULTICLASSMACHINE_H___
#define _NATIVEMULTICLASSMACHINE_H___

#include <machine/MulticlassMachine.h>

namespace shogun
{

/** @brief experimental abstract native multiclass machine class */
class CNativeMulticlassMachine : public CMulticlassMachine
{
	public:
		/** default constructor  */
		CNativeMulticlassMachine()
		{
		}

		/** destructor */
		virtual ~CNativeMulticlassMachine()
		{
		}

		/** get name */
		virtual const char* get_name() const
		{
			return "NativeMulticlassMachine";
		}

	protected:
		/** init strategy */
		void init_strategy() { }

		/** clear machines */
		void clear_machines() { }

		/** abstract init machine for training method */
		virtual bool init_machine_for_train(CFeatures* data) { return true; }

		/** abstract init machines for applying method */
		virtual bool init_machines_for_apply(CFeatures* data) { return true; }

		/** check whether machine is ready */
		virtual bool is_ready() { return true; }

		/** obtain machine from trained one */
		virtual CMachine* get_machine_from_trained(CMachine* machine) { return NULL; }

		/** get num rhs vectors */
		virtual int32_t get_num_rhs_vectors() { return 0; }

		/** set subset to the features of the machine, deletes old one
		 *
		 * @param subset subset indices to set
		 */
		virtual void add_machine_subset(SGVector<index_t> subset) { }

		/** deletes any subset set to the features of the machine */
		virtual void remove_machine_subset() { }

		/** whether the machine is acceptable in set_machine */
		virtual bool is_acceptable_machine(CMachine *machine) { return true; }

};
}
#endif
