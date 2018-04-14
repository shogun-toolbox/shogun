/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */
#ifndef __DESERIALIZER_H__
#define __DESERIALIZER_H__

#include <shogun/base/SGObject.h>
#include <shogun/io/stream/InputStream.h>

namespace shogun
{
	class CDeserializer : public CSGObject
	{
	public:
		CDeserializer();
		virtual ~CDeserializer();
		virtual void attach(Some<CInputStream> stream) = 0;
		virtual Some<CSGObject> read() = 0;

	private:
		Some<CInputStream> m_stream;
	};
}

#endif
