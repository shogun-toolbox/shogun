/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */
#include <shogun/lib/config.h>
#ifdef HAVE_TFLOGGER

#ifndef SHOGUN_PARAMETEROBSERVERTENSORBOARD_H
#define SHOGUN_PARAMETEROBSERVERTENSORBOARD_H

#include <shogun/lib/observers/ParameterObserver.h>
#include <tflogger/event_logger.h>

namespace shogun
{
	class ParameterObserverTensorBoard : public ParameterObserver
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
		ParameterObserverTensorBoard(
		    std::vector<std::string>& parameters,
		    std::vector<ParameterProperties>& properties);

		/**
		 * Constructor
		 * @param filename name of the generated output file
		 * @param parameters list of parameters which we want to watch over
		 */
		ParameterObserverTensorBoard(
		    const std::string& filename, std::vector<std::string>& parameters,
		    std::vector<ParameterProperties>& properties);

		ParameterObserverTensorBoard(std::vector<std::string>& parameters);
		ParameterObserverTensorBoard(
		    std::vector<ParameterProperties>& properties);

		/**
		 * Virtual destructor
		 */
		virtual ~ParameterObserverTensorBoard();

	protected:
		/**
		* Writer object which will be used to write tensorflow::Event files
		*/
		tflogger::EventLogger m_writer;
	};
}

#endif // SHOGUN_PARAMETEROBSERVERTENSORBOARD_H
#endif // HAVE_TFLOGGER
