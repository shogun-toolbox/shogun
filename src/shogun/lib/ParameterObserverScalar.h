/*
* Written (W) 2017 Giovanni De Toni
*/

#ifndef SHOGUN_PARAMETEROBSERVERSCALAR_H
#define SHOGUN_PARAMETEROBSERVERSCALAR_H

#include <shogun/lib/ParameterObserverInterface.h>

namespace shogun
{
	/**
	 * Implementation of a ParameterObserver which write to file
	 * scalar values, given object emitted from a parameter observable.
	 */
	class ParameterObserverScalar : public ParameterObserverInterface
	{

	public:
		ParameterObserverScalar();
		ParameterObserverScalar(std::vector<std::string>& parameters);
		ParameterObserverScalar(
		    const std::string& filename, std::vector<std::string>& parameters);
		~ParameterObserverScalar();

		virtual bool filter(const std::string& param);

		virtual void on_next(const ObservedValue& value);
		virtual void on_error(std::exception_ptr);
		virtual void on_complete();
	};
}

#endif // SHOGUN_PARAMETEROBSERVERSCALAR_H
