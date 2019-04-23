/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */
#ifndef SHOGUN_PARAMETEROBSERVERLOGGER_H
#define SHOGUN_PARAMETEROBSERVERLOGGER_H

#include <shogun/lib/observers/ParameterObserver.h>

namespace shogun
{

	/**
	 * This class implements a logger which prints all observed updates.
	 */
	class ParameterObserverLogger : public ParameterObserver
	{

	public:
		ParameterObserverLogger();

		ParameterObserverLogger(std::vector<std::string>& parameters);

		virtual ~ParameterObserverLogger();

		virtual void on_error(std::exception_ptr ptr);

		virtual void on_complete();

		virtual const char* get_name() const
		{
			return "ParameterObserverLogger";
		};

	protected:
		virtual void on_next_impl(const TimedObservedValue& value);
	};
}

#endif // SHOGUN_PARAMETEROBSERVERLOGGER_H
