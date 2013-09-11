/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef _LINEARMULTICLASSMACHINE_H___
#define _LINEARMULTICLASSMACHINE_H___

#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/MulticlassMachine.h>

namespace shogun
{

class CDotFeatures;
class CLinearMachine;
class CMulticlassStrategy;

/** @brief generic linear multiclass machine */
class CLinearMulticlassMachine : public CMulticlassMachine
{
	public:
		/** default constructor  */
		CLinearMulticlassMachine() : CMulticlassMachine(), m_features(NULL)
		{
			SG_ADD((CSGObject**)&m_features, "m_features", "Feature object.",
			    MS_NOT_AVAILABLE);
		}

		/** standard constructor
		 * @param strategy multiclass strategy
		 * @param features features
		 * @param machine linear machine
		 * @param labs labels
		 */
		CLinearMulticlassMachine(CMulticlassStrategy *strategy, CDotFeatures* features, CLinearMachine* machine, CLabels* labs) :
			CMulticlassMachine(strategy,(CMachine*)machine,labs), m_features(NULL)
		{
			set_features(features);
			SG_ADD((CSGObject**)&m_features, "m_features", "Feature object.",
			    MS_NOT_AVAILABLE);
		}

		/** destructor */
		virtual ~CLinearMulticlassMachine()
		{
			SG_UNREF(m_features);
		}

		/** get name */
		virtual const char* get_name() const
		{
			return "LinearMulticlassMachine";
		}

		/** set features
		 *
		 * @param f features
		 */
		void set_features(CDotFeatures* f)
		{
			SG_REF(f);
			SG_UNREF(m_features);
			m_features = f;

			for (index_t i=0; i<m_machines->get_num_elements(); i++)
				((CLinearMachine* )m_machines->get_element(i))->set_features(f);
		}

		/** get features
		 *
		 * @return features
		 */
		CDotFeatures* get_features() const
		{
			SG_REF(m_features);
			return m_features;
		}

	protected:

		/** init machine for train with setting features */
		virtual bool init_machine_for_train(CFeatures* data)
		{
			if (!m_machine)
				SG_ERROR("No machine given in Multiclass constructor\n")

			if (data)
				set_features((CDotFeatures*)data);

			((CLinearMachine*)m_machine)->set_features(m_features);

			return true;
		}

		/** init machines for applying with setting features */
		virtual bool init_machines_for_apply(CFeatures* data)
		{
			if (data)
				set_features((CDotFeatures*)data);

			for (int32_t i=0; i<m_machines->get_num_elements(); i++)
			{
				CLinearMachine* machine = (CLinearMachine*)m_machines->get_element(i);
				ASSERT(m_features)
				ASSERT(machine)
				machine->set_features(m_features);
				SG_UNREF(machine);
			}

			return true;
		}

		/** check features availability */
		virtual bool is_ready()
		{
			if (m_features)
				return true;

			return false;
		}

		/** construct linear machine from given linear machine */
		virtual CMachine* get_machine_from_trained(CMachine* machine)
		{
			return new CLinearMachine((CLinearMachine*)machine);
		}

		/** get number of rhs feature vectors */
		virtual int32_t get_num_rhs_vectors()
		{
			return m_features->get_num_vectors();
		}

		/** set subset to the features of the machine, deletes old one
		 *
		 * @param subset subset instance to set
		 */
		virtual void add_machine_subset(SGVector<index_t> subset)
		{
			/* changing the subset structure to use subset stacks. This might
			 * have to be revised. Heiko Strathmann */
			m_features->add_subset(subset);
		}

		/** deletes any subset set to the features of the machine */
		virtual void remove_machine_subset()
		{
			/* changing the subset structure to use subset stacks. This might
			 * have to be revised. Heiko Strathmann */
			m_features->remove_subset();
		}

		/** Stores feature data of underlying model. Does nothing because
		 * Linear machines store the normal vector of the separating hyperplane
		 * and therefore the model anyway
		 */
		virtual void store_model_features() {}

	protected:

		/** features */
		CDotFeatures* m_features;
};
}
#endif
