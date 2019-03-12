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

namespace shogun {

	/**
	 * Template ObservedValue which is used to store the real value we
	 * want to send to the parameter observers. Therefore, we will be able
	 * to expose only the ObservedValue class to SWIG without worrying
	 * about the underlining type.
	 * @tparam T the type of the observed value
	 */
	template<class T>
	class ObservedValueTemplated : public ObservedValue {

	public:

		/**
		 * Constructor
		 * @param step step
		 * @param name the observed value's name
		 * @param value the observed value
		 */
		ObservedValueTemplated(int64_t step, std::string name, T value)
				: ObservedValue(step, name), m_observed_value(value) {
			this->watch_param("value", &m_observed_value,
							  AnyParameterProperties("Value of the observation", ParameterProperties::NONE));
		}

		/**
		 * Destructor
		 */
		~ObservedValueTemplated() {};

		/**
		 * @copydoc ObservedValue::get_any()
		 * This method returns an Any reference of the observed
		 * value stored by this class.
		 */
		virtual Any get_any() {
			return make_any(m_observed_value);
		}

	private:
		/**
		 * Templated observed value
		 */
		T m_observed_value;
	};
}

#endif //SHOGUN_OBSERVEDVALUETEMPLATED_H
