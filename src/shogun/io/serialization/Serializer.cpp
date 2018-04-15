/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#include <shogun/io/serialization/Serializer.h>

using namespace shogun;
using namespace shogun::io;

CSerializer::CSerializer() : CSGObject(), m_stream(empty<COutputStream>())
{
}

CSerializer::~CSerializer()
{
}

void CSerializer::attach(Some<COutputStream> stream)
{
    m_stream = stream;
}

Some<COutputStream> CSerializer::stream() const
{
	REQUIRE(m_stream, "Serializer has no stream, attach() it to a stream");
	return m_stream;
}
