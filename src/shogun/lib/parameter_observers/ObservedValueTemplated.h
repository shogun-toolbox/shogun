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
		ObservedValueTemplated(int64_t step, char * name, T value)
				: ObservedValue(step, name) {
			SG_ADD(&v, "value", "Value of the observation", ParameterProperties::NONE);
		}

		~ObservedValueTemplated() {};

		Any get_any() {
			return make_any(v);
		}

	private:
		T v;
	};
}

#endif //SHOGUN_OBSERVEDVALUETEMPLATED_H
