/*
* Written (W) 2017 Giovanni De Toni
*/
#include <shogun/lib/config.h>
#ifdef HAVE_TFLOGGER

#ifndef SHOGUN_OUTPUTFORMAT_H
#define SHOGUN_OUTPUTFORMAT_H

#include <shogun/base/SGObject.h>
#include <shogun/lib/any.h>
#include <tflogger/event.pb.h>

#include <utility>

namespace shogun
{
	/**
	 * Convert an std::pair<std::string, Any> to a tensorflow::Event,
	 * which can be written to file and used with tools like Tensorboard.
	 */
	class TBOutputFormat : public CSGObject
	{

	public:
		TBOutputFormat();
		~TBOutputFormat();

		/**
		 * Generate a tensorflow::Event object give some informations
		 * @param event_step the current event step
		 * @param value the value which will be converted to tensorflow::Event
		 * @param node_name the node name (default: node)
		 * @return the newly created tensorflow::Event
		 */
		tensorflow::Event convert_scalar(
		    const int64_t& event_step, const std::pair<std::string, Any>& value,
		    std::string& node_name);

		tensorflow::Event convert_vector(
			const int64_t& event_step, const std::pair<std::string, Any>& value,
			std::string& node_name);

		virtual const char * get_name() const
		{
			return "TFLogger";
		}
	};
}

#endif // SHOGUN_OUTPUTFORMAT_H
#endif // HAVE_TFLOGGER
