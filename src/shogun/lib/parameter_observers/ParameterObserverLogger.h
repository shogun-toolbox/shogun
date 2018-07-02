/*
* Written (W) 2018 Giovanni De Toni
*/

#include <shogun/lib/config.h>

#ifndef SHOGUN_PARAMETEROBSERVERLOGGER_H
#define SHOGUN_PARAMETEROBSERVERLOGGER_H

#include <shogun/lib/parameter_observers/ParameterObserverInterface.h>
#include <shogun/base/SGObject.h>

namespace shogun {

	class CParameterObserverLogger : public ParameterObserverInterface, public CSGObject {
	public:

		CParameterObserverLogger();
		~CParameterObserverLogger();

		virtual void on_next(const TimedObservedValue& value);
		virtual void on_error(std::exception_ptr ptr);
		virtual void on_complete();

		virtual const char* get_name() const
		{
			return "ParameterObserverLogger";
		}
	};
}


#endif //SHOGUN_PARAMETEROBSERVERLOGGER_H
