/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#include <shogun/io/serialization/JsonSerializer.h>

#include <rapidjson/writer.h>

using namespace shogun;

struct COutputStreamAdapter
{
	typedef char Ch;
	void Put(Ch c)
	{
		m_stream->write(&c, 1);
	}

	void Flush()
	{
		m_stream->flush();
	}

	COutputStream* m_stream;
};

CJsonSerializer::CJsonSerializer() : CSerializer()
{
}

CJsonSerializer::~CJsonSerializer()
{
}

void CJsonSerializer::write(Some<CSGObject> object)
{
	COutputStreamAdapter adapter{ .m_stream = stream().get() };
	rapidjson::Writer<COutputStreamAdapter> writer(adapter);
	writer.StartObject();
	writer.Key("name");
	writer.String(object->get_name());
	//writer.String(object.get());
	writer.EndObject();
}
