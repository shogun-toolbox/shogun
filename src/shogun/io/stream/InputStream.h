/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */
#ifndef __INPUT_STREAM_H__
#define __INPUT_STREAM_H__

#include <shogun/base/SGObject.h>

namespace shogun
{
	class CInputStream : public CSGObject
	{
	public:
		CInputStream();
		virtual ~CInputStream();

		virtual void read(void* buffer, size_t size) = 0;
		virtual char read() = 0;
	};
}

#endif
