/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */
#ifndef __INPUT_STREAM_H__
#define __INPUT_STREAM_H__

#include <shogun/base/SGObject.h>

#include <string>

namespace shogun
{
	namespace io
	{
		class CInputStream : public CSGObject
		{
		public:
			CInputStream();
			virtual ~CInputStream();

			virtual std::error_condition read(std::string* buffer, int64_t size) = 0;
			virtual std::error_condition skip(int64_t bytes) = 0;
			virtual int64_t tell() const = 0;
			virtual void reset() = 0;
		};
	}
}

#endif
