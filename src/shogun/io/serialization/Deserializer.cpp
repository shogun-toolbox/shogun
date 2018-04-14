/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#include <shogun/io/serialization/Deserializer.h>

using namespace shogun;

CDeserializer::CDeserializer() : m_stream(nullptr)
{
}

CDeserializer::~CDeserializer()
{
}

void CDeserializer::attach(Some<CInputStream> stream)
{
	m_stream = stream;
}
