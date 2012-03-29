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
#include <shogun/base/Parameter.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/MulticlassMachine.h>

namespace shogun
{

class Parameter;
class CDotFeatures;
class CLinearMachine;

/** @brief generic linear multiclass machine */
class CLinearMulticlassMachine : public CMulticlassMachine
{
	public:
		/** default constructor  */
		CLinearMulticlassMachine() : CMulticlassMachine(), m_features(NULL)
		{
			m_parameters->add((CSGObject**)&m_features,"m_features");
		}

		/** standard constructor
		 * @param strategy multiclass strategy
		 * @param features features 
		 * @param machine linear machine
		 * @param labs labels
		 */
		CLinearMulticlassMachine(EMulticlassStrategy strategy, CDotFeatures* features, CLinearMachine* machine, CLabels* labs) :
			CMulticlassMachine(strategy,(CMachine*)machine,labs), m_features(NULL)
		{
			set_features(features);
			m_parameters->add((CSGObject**)&m_features,"m_features");
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
		}

		/** get features
		 *
		 * @return kernel
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
				SG_ERROR("No machine given in Multiclass constructor\n");

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

			for (int32_t i=0; i<m_machines.vlen; i++)
				((CLinearMachine*)m_machines[i])->set_features(m_features);

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
		virtual void set_machine_subset(CSubset* subset)
		{
			m_features->set_subset(subset);
		}

		/** deletes any subset set to the features of the machine */
		virtual void remove_machine_subset()
		{
			m_features->remove_subset();
		}

	protected:

		/** features */
		CDotFeatures* m_features;

};
}
#endif
