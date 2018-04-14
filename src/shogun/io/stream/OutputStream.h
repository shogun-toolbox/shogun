/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */
#ifndef __OUTPUT_STREAM_H__
#define __OUTPUT_STREAM_H__

#include <shogun/base/SGObject.h>

namespace shogun
{
	class COutputStream : public CSGObject
	{
	public:
		COutputStream();
		virtual ~COutputStream();

		virtual void write(void* buffer, size_t size) = 0;
	};
}

#endif
