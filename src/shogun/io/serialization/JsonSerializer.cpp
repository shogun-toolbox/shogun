/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Viktor Gal
 */

#include <memory>

#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

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

template<typename Writer> void object_writer(Writer& writer, Some<CSGObject> object);

template<typename RapidJsonWriter>
class JSONWriterVisitor : public AnyVisitor
{
public:
	JSONWriterVisitor(RapidJsonWriter& jw):
		AnyVisitor(), m_json_writer(jw) {}

	virtual void on(bool* v)
	{
		SG_SDEBUG("writing bool with value %d\n", *v)
		m_json_writer.Bool(*v);
	}
	virtual void on(int32_t* v)
	{
		SG_SDEBUG("writing int32_t with value %d\n", *v)
		m_json_writer.Int(*v);
	}
	virtual void on(int64_t* v)
	{
		SG_SDEBUG("writing int64_t with value %d\n", *v)
		m_json_writer.Int64(*v);
	}
	virtual void on(float* v)
	{
		SG_SDEBUG("writing float with value %f\n", *v)
		m_json_writer.Double(*v);
	}
	virtual void on(double* v)
	{
		SG_SDEBUG("writing double with value %f\n", *v)
		m_json_writer.Double(*v);
	}
	virtual void on(CSGObject** v)
	{
		if (*v)
		{
			SG_SDEBUG("writing SGObject with of type\n")
			object_writer(m_json_writer, wrap<CSGObject>(*v));
		}
	}
	virtual void on(SGVector<int>* v)
	{
		SG_SDEBUG("writing SGVector<int>\n")
		m_json_writer.StartArray();
		for (const auto& i: *v)
			m_json_writer.Int(i);
		m_json_writer.EndArray();
	}
	virtual void on(SGVector<float>* v)
	{
		SG_SDEBUG("writing SGVector<float>\n")
	}
	virtual void on(SGVector<double>* v)
	{
		SG_SDEBUG("writing SGVector<double>\n")

	}
	virtual void on(SGMatrix<int>* v)
	{
		SG_SDEBUG("writing SGMatrix<int>\n")

	}
	virtual void on(SGMatrix<float>* v)
	{
		SG_SDEBUG("writing SGMatrix<float>\n")

	}
	virtual void on(SGMatrix<double>* v)
	{
		SG_SDEBUG("writing SGMatrix<double>\n")

	}

private:
	RapidJsonWriter& m_json_writer;
};

template<typename Writer>
void object_writer(Writer& writer, Some<CSGObject> object)
{
	auto writer_visitor = std::make_unique<JSONWriterVisitor<Writer>>(writer);
	writer.Key("name");
	writer.String(object->get_name());
	writer.Key("generic");
	writer.Int(object->get_generic());
	auto param_names = object->parameter_names();
	writer.Key("parameters");
	writer.StartObject();
	for (auto param_name: param_names)
	{
		writer.Key(param_name.c_str());
		BaseTag tag(param_name);
		auto param = object->get_parameter(tag);
		param.get_value().visit(writer_visitor.get());
	}
	writer.EndObject();
}

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
	object_writer(writer, object);
}
