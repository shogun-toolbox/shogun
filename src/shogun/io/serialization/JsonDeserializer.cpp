/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#include <shogun/io/serialization/JsonDeserializer.h>

using namespace shogun;

CJsonDeserializer::CJsonDeserializer() : CDeserializer()
{
}

CJsonDeserializer::~CJsonDeserializer()
{
}

Some<CSGObject> CJsonDeserializer::read()
{
	return wrap<CSGObject>(nullptr);
}
