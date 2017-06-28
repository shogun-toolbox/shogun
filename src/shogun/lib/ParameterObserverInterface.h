#ifndef SHOGUN_PARAMETEROBSERVERINTERFACE_H
#define SHOGUN_PARAMETEROBSERVERINTERFACE_H

#include <stdexcept>
#include <utility>
#include <vector>

#include <rxcpp/rx-observable.hpp>
#include <shogun/lib/any.h>
#include <tflogger/tensorflow_logger.h>

namespace shogun
{
	/**
	 * Interface for the parameter observer classes
	 */
	class ParameterObserverInterface
	{

	public:

		/* One observed value, composed of:
		*  - step (for the graph x axis);
		*  - a pair composed of: parameter's name + parameter's value
		*/
		typedef std::pair<int64_t, std::pair<std::string, Any>> ObservedValue;

		/**
		* Default constructor
		*/
		ParameterObserverInterface();

		/**
		 * Constructor
		 * @param parameters list of parameters which we want to watch over
		 */
		ParameterObserverInterface(std::vector<std::string>& parameters);

		/**
		 * Constructor
		 * @param filename name of the generated output file
		 * @param parameters list of parameters which we want to watch over
		 */
		ParameterObserverInterface(
		    const std::string& filename, std::vector<std::string>& parameters);
		/**
		 * Virtual destructor
		 */
		virtual ~ParameterObserverInterface();

		/**
		 * Filter function, check if the parameter name supplied is what
		 * we want to monitor
		 * @param param the param name
		 * @return true if param is found inside of m_parameters list
		 */
		virtual bool filter(const std::string& param) = 0;

		/**
		 * Method which will be called when the parameter observable emits a
		 * value.
		 * @param value the value emitted by the parameter observable
		 */
		virtual void on_next(const ObservedValue& value) = 0;
		/**
		 * Method which will be called on errors
		 */
		virtual void on_error(std::exception_ptr) = 0;
		/**
		 * Method which will be called on completion
		 */
		virtual void on_complete() = 0;

	protected:
		/**
		 * List of parameter's names we want to monitor
		 */
		std::vector<std::string> m_parameters;
		/**
		 * Writer object which will be used to write tensorflow::Event files
		 */
		tflogger::TensorFlowLogger m_writer;
	};
}

#endif // SHOGUN_PARAMETEROBSERVER_H
