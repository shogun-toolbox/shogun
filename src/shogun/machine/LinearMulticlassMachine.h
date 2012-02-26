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
		 * @param machine machine
		 * @param labels labels
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

		virtual bool init_machine_for_train(CFeatures* data)
		{
			if (data)
				set_features((CDotFeatures*)data);

			((CLinearMachine*)m_machine)->set_features(m_features);

			return true;
		}

		virtual bool init_machines_for_apply(CFeatures* data)
		{
			if (data)
				set_features((CDotFeatures*)data);

			for (int32_t i=0; i<m_machines.vlen; i++)
				((CLinearMachine*)m_machines[i])->set_features(m_features);

			return true;
		}

		virtual bool is_ready()
		{
			if (m_features)
					return true;

			return false;
		}

		virtual CMachine* get_machine_from_trained(CMachine* machine)
		{
			return new CLinearMachine((CLinearMachine*)machine);
		}

		virtual int32_t get_num_rhs_vectors()
		{
			return m_features->get_num_vectors();
		}

	private:

		CDotFeatures* m_features;

};
}
#endif
