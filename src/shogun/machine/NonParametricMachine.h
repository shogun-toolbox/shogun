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
		NonParametricMachine() : Machine()
		{
			// TODO : when all refactor is done, m_labels should be removed from
			// Machine Class
			SG_ADD(
			    &m_labels, "labels", "labels used in train machine algorithm");
			SG_ADD(
			    &m_features, "features_train",
			    "Training features of nonparametric model",
			    ParameterProperties::READONLY);
		}
		virtual ~NonParametricMachine()
		{
		}
		using Machine::train;

		bool train(
		    const std::shared_ptr<Features>& data,
		    const std::shared_ptr<Labels>& lab) override
		{
			m_labels = lab;
			require(
				data->get_num_vectors() == m_labels->get_num_labels(),
				"Number of training vectors ({}) does not match number of "
				"labels ({})", 
				data->get_num_vectors(), m_labels->get_num_labels());
			return Machine::train(data);
		}

		const char* get_name() const override
		{
			return "NonParametricMachine";
		}
		
        virtual void set_labels(std::shared_ptr<Labels> lab)
		{
			m_labels = lab;
		}

        /** get labels
         *
         * @return labels
         */
        virtual std::shared_ptr<Labels> get_labels()
		{
			return m_labels;
		}
	protected:
		std::shared_ptr<Features> m_features;
		std::shared_ptr<Labels> m_labels;
	};
} // namespace shogun
#endif