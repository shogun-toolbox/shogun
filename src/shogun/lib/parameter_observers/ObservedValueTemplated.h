/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */

#ifndef SHOGUN_OBSERVEDVALUETEMPLATED_H
#define SHOGUN_OBSERVEDVALUETEMPLATED_H

#include <shogun/base/Parameter.h>
#include <shogun/lib/parameter_observers/ObservedValue.h>

namespace shogun
{

	/**
	 * Templated specialisation of ObservedValue that stores the actual data.
	 * @tparam T the type of the observed value
	 */
	template <class T>
	class ObservedValueTemplated : public ObservedValue
	{

	public:
		/**
		 * Constructor
		 * @param step step
		 * @param name the observed value's name
		 * @param value the observed value
		 */
		ObservedValueTemplated(int64_t step, std::string name, std::string description, T value)
		    : ObservedValue(step, name), m_observed_value(value)
		{
			this->watch_param(
			    "value", &m_observed_value,
			    AnyParameterProperties(
			        description, ParameterProperties::NONE));
            m_any_value = make_any(m_observed_value);
		}

		/**
		 * Destructor
		 */
		~ObservedValueTemplated(){};

	private:
		/**
		 * Templated observed value
		 */
		T m_observed_value;
	};
}

#endif // SHOGUN_OBSERVEDVALUETEMPLATED_H
