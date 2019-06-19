/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */
#ifndef __OUTPUT_STREAM_H__
#define __OUTPUT_STREAM_H__

#include <shogun/base/SGObject.h>

namespace shogun
{
	namespace io
	{
		class COutputStream : public CSGObject
		{
		public:
			COutputStream();
			virtual ~COutputStream();

			virtual std::error_condition close() = 0;
			virtual std::error_condition flush() = 0;
			virtual std::error_condition write(const void* buffer, int64_t size) = 0;
		};
	}
}

#endif
