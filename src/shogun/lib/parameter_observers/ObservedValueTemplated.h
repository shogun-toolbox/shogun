/*
* Written (W) 2019 Giovanni De Toni
*/

#ifndef SHOGUN_OBSERVEDVALUETEMPLATED_H
#define SHOGUN_OBSERVEDVALUETEMPLATED_H

#include <shogun/base/Parameter.h>
#include <shogun/lib/parameter_observers/ObservedValue.h>

namespace shogun {

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
			SG_ADD(&observed_value, "value", "Value of the observation", ParameterProperties::NONE);
		}

		~ObservedValueTemplated() {};

		Any get_any() {
			return make_any(v);
		}

	private:
		T m_observed_value;
	};
}

#endif //SHOGUN_OBSERVEDVALUETEMPLATED_H
