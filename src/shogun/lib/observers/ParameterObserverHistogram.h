/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */
#include <shogun/lib/config.h>
#ifdef HAVE_TFLOGGER

#ifndef SHOGUN_PARAMETEROBSERVERHISTOGRAM_H
#define SHOGUN_PARAMETEROBSERVERHISTOGRAM_H

#include <shogun/base/SGObject.h>
#include <shogun/lib/observers/ParameterObserverTensorBoard.h>

namespace shogun
{
	/**
	 * Implementation of a ParameterObserver which write to file
	 * histograms, given object emitted from a parameter observable.
	 */
	class ParameterObserverHistogram : public ParameterObserverTensorBoard
	{

	public:
		ParameterObserverHistogram();

		ParameterObserverHistogram(std::vector<std::string>& parameters);

		ParameterObserverHistogram(
		    std::vector<ParameterProperties>& properties);

		ParameterObserverHistogram(
		    std::vector<std::string>& parameters,
		    std::vector<ParameterProperties>& properties);

		ParameterObserverHistogram(
		    const std::string& filename, std::vector<std::string>& parameters,
		    std::vector<ParameterProperties>& properties);

		~ParameterObserverHistogram();

		virtual void on_error(std::exception_ptr);
		virtual void on_complete();

		/**
		* Get class name.
		* @return class name
		*/
		virtual const char* get_name() const
		{
			return "ParameterObserverHistogram";
		}

	protected:
		virtual void on_next_impl(const TimedObservedValue& value);
	};
}

#endif // SHOGUN_PARAMETEROBSERVERHISTOGRAM_H
#endif // HAVE_TFLOGGER
