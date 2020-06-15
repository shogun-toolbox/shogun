/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */

#ifndef NONPARAMETRCMACHINE_H_
#define NONPARAMETRCMACHINE_H_

#include <shogun/machine/Machine.h>

namespace shogun
{

	class NonParametricMachine : public Machine
	{
	public:
		NonParametricMachine(): Machine()
		{
			//TODO : when all refactor is done, m_labels should be removed from 
            //Machine Class
			// SG_ADD(
			//     &m_labels, "labels", "labels used in train machine algorithm",
			//     ParameterProperties::READONLY);
			SG_ADD(&m_features, "features_train",
			    "Training features of nonparametric model",
			    ParameterProperties::READONLY);
		}
		virtual ~NonParametricMachine()
		{
		}

		const char* get_name() const override{ return "NonParametricMachine"; }

	protected:
		
		std::shared_ptr<Features> m_features;

        //TODO
		// when all refactor is done, we should use this m_labels
		// std::shared_ptr<Labels> m_labels;
	};
} // namespace shogun
#endif