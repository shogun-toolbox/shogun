/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#include <shogun/io/serialization/Deserializer.h>

using namespace shogun;
using namespace shogun::io;

CDeserializer::CDeserializer() : CSGObject(), m_stream(empty<CInputStream>())
{
}

CDeserializer::~CDeserializer()
{
}

void CDeserializer::attach(Some<CInputStream> stream)
{
	m_stream = stream;
}

Some<CInputStream> CDeserializer::stream() const
{
	REQUIRE(m_stream, "Deserializer has no stream, attach() it to a stream");
	return m_stream;
}
