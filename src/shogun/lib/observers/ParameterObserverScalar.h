/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */
#include <shogun/lib/config.h>
#ifdef HAVE_TFLOGGER

#ifndef SHOGUN_PARAMETEROBSERVERSCALAR_H
#define SHOGUN_PARAMETEROBSERVERSCALAR_H

#include <shogun/base/SGObject.h>
#include <shogun/lib/observers/ParameterObserverTensorBoard.h>

namespace shogun
{
	/**
	 * Implementation of a ParameterObserver which write to file
	 * scalar values, given object emitted from a parameter observable.
	 */
	class ParameterObserverScalar : public ParameterObserverTensorBoard
	{

	public:
		ParameterObserverScalar();

        ParameterObserverScalar(std::vector<std::string> &parameters);

        ParameterObserverScalar(std::vector<ParameterProperties> &properties);

        ParameterObserverScalar(
            std::vector<std::string>& parameters,
            std::vector<ParameterProperties>& properties);

        ParameterObserverScalar(
		    const std::string& filename,
            std::vector<std::string>& parameters,
            std::vector<ParameterProperties>& properties);

        ~ParameterObserverScalar();

		virtual void on_error(std::exception_ptr);
		virtual void on_complete();

		/**
		* Get class name.
		* @return class name
		*/
		virtual const char* get_name() const
		{
			return "ParameterObserverScalar";
		}

	protected:
		virtual void on_next_impl(const TimedObservedValue& value);
	};
}

#endif // SHOGUN_PARAMETEROBSERVERSCALAR_H
#endif // HAVE_TFLOGGER
