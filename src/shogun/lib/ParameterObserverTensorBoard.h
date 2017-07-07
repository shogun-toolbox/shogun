/*
* Written (W) 2017 Giovanni De Toni
*/

#ifndef SHOGUN_PARAMETEROBSERVERTENSORBOARD_H
#define SHOGUN_PARAMETEROBSERVERTENSORBOARD_H

#include <shogun/lib/ParameterObserverInterface.h>

#include <tflogger/tensorflow_logger.h>

namespace shogun
{
	class ParameterObserverTensorBoard : public ParameterObserverInterface
	{

	public:
		/**
		* Default constructor
		*/
		ParameterObserverTensorBoard();

		/**
		 * Constructor
		 * @param parameters list of parameters which we want to watch over
		 */
		ParameterObserverTensorBoard(std::vector<std::string>& parameters);

		/**
		 * Constructor
		 * @param filename name of the generated output file
		 * @param parameters list of parameters which we want to watch over
		 */
		ParameterObserverTensorBoard(
		    const std::string& filename, std::vector<std::string>& parameters);
		/**
		 * Virtual destructor
		 */
		virtual ~ParameterObserverTensorBoard();

	protected:
		/**
		* Writer object which will be used to write tensorflow::Event files
		*/
		tflogger::TensorFlowLogger m_writer;
	};
}

#endif // SHOGUN_PARAMETEROBSERVERTENSORBOARD_H
