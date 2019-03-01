/*
* Written (W) 2019 Giovanni De Toni
*/

#ifndef SHOGUN_PARAMETEROBSERVERLOGGER_H
#define SHOGUN_PARAMETEROBSERVERLOGGER_H

#include <shogun/base/SGObject.h>
#include <shogun/lib/parameter_observers/ParameterObserverInterface.h>

namespace shogun {
	/**
	 * This class implements a logger which will basically print on screen all
	 * the information received through the observable infrastructure.
	 */
	class ParameterObserverLogger : public ParameterObserverInterface, public CSGObject {

	public:
		ParameterObserverLogger();
		ParameterObserverLogger(std::vector<std::string> &parameters);

		/**
		 * This will print on screen everything it receives
		 * @param value the observed value
		 */
		virtual void on_next(const TimedObservedValue &value);

		virtual void on_error(std::exception_ptr ptr);

		virtual void on_complete();

		const char* get_name() const {
			return "ParameterObserverLogger";
		}

	};

}

#endif //SHOGUN_PARAMETEROBSERVERLOGGER_H
