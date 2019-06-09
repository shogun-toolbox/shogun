/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 */

#ifndef SHOGUN_OBSERVEDVALUETEMPLATED_H
#define SHOGUN_OBSERVEDVALUETEMPLATED_H

#include <shogun/lib/observers/ObservedValue.h>

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
		ObservedValueTemplated(
		    const int64_t step, const std::string& name,
		    const std::string& description, const T value)
		    : ObservedValue(step, name), m_observed_value(value)
		{
			this->watch_param(
			    name, &m_observed_value,
			    AnyParameterProperties(
			        description, ParameterProperties::READONLY));
			m_any_value = make_any(m_observed_value);
		}

		/**
		 * Constructor which takes AnyParameterProperties for the observed value
		 * @param step step
		 * @param name the observed value's name
		 * @param value the observed value
		 * @param properties properties of that observed value
		 */
		ObservedValueTemplated(
		    const int64_t step, const std::string& name, const T value,
		    const AnyParameterProperties properties)
		    : ObservedValue(step, name), m_observed_value(value)
		{
			this->watch_param(name, &m_observed_value, properties);
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
