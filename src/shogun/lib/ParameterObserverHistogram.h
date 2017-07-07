/*
* Written (W) 2017 Giovanni De Toni
*/
#include <shogun/lib/config.h>
#ifdef HAVE_TFLOGGER

#ifndef SHOGUN_PARAMETEROBSERVERHISTOGRAM_H
#define SHOGUN_PARAMETEROBSERVERHISTOGRAM_H

#include <shogun/lib/ParameterObserverTensorBoard.h>

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
		    const std::string& filename, std::vector<std::string>& parameters);
		~ParameterObserverHistogram();

		virtual bool filter(const std::string& param);

		virtual void on_next(const ObservedValue& value);
		virtual void on_error(std::exception_ptr);
		virtual void on_complete();
	};
}

#endif // SHOGUN_PARAMETEROBSERVERHISTOGRAM_H
#endif // HAVE_TFLOGGER
