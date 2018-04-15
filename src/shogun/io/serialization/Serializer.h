/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */
#ifndef __SERIALIZER_H__
#define __SERIALIZER_H__

#include <shogun/base/SGObject.h>
#include <shogun/io/stream/OutputStream.h>

namespace shogun
{
	class CSerializer : public CSGObject
	{
	public:
		CSerializer();
		virtual ~CSerializer();
		virtual void attach(Some<COutputStream> stream)
		virtual void write(Some<CSGObject> object) = 0;
		Some<COutputStream> stream() const;

	private:
		Some<COutputStream> m_stream;
	};
}

#endif
