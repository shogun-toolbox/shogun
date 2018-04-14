/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#include <shogun/io/serialization/Serializer.h>

using namespace shogun;

CSerializer::CSerializer() : m_stream(nullptr)
{
}

CSerializer::~CSerializer()
{
}

void CSerializer::attach(Some<COutputStream> stream)
{
	m_stream = stream;
}
