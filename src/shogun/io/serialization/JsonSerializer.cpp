/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Viktor Gal
 */

#include <memory>

#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

#include <rapidjson/writer.h>

using namespace rapidjson;
using namespace shogun;
using namespace shogun::io;
using namespace std;

extern const char* const kNameKey;
const char* const kNameKey = "name";
extern const char* const kGenericKey;
const char* const kGenericKey = "generic";
extern const char* const kParametersKey;
const char* const kParametersKey = "parameters";

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

	Some<COutputStream> m_stream;
};

template<typename Writer> void write_object(Writer& writer, Some<CSGObject> object);

template<typename Writer, typename Fn, typename V>
void write_vector(Writer& writer, Fn f, V array)
{
	writer.StartArray();
	for (const auto& v: *array)
	{
		if (!(writer.*f)(v))
			throw runtime_error("Could not serializer vector value!");
	}
	writer.EndArray();
}

template<typename Writer, typename Fn, typename V>
void write_matrix(Writer& writer, Fn f, V m)
{
	writer.StartArray();
	int64_t ctr = 0;
	writer.StartArray();
	for (const auto& v: *m)
	{
		if (!(writer.*f)(v))
			throw runtime_error("Could not serializer matrix value!");
		if (++ctr % m->num_rows == 0)
		{
			writer.EndArray();
			if (ctr < m->num_cols*m->num_rows)
				writer.StartArray();
		}
	}
	writer.EndArray();
}

template<class Writer>
class JSONWriterVisitor : public AnyVisitor
{
public:
	JSONWriterVisitor(Writer& jw):
		AnyVisitor(), m_json_writer(jw) {}

	virtual void on(bool* v)
	{
		SG_SDEBUG("writing bool with value %d\n", *v);
		m_json_writer.Bool(*v);
	}
	virtual void on(int32_t* v)
	{
		SG_SDEBUG("writing int32_t with value %d\n", *v);
		m_json_writer.Int(*v);
	}
	virtual void on(int64_t* v)
	{
		SG_SDEBUG("writing int64_t with value %" PRId64 "\n", *v);
		m_json_writer.Int64(*v);
	}
	virtual void on(uint64_t* v)
	{
		SG_SDEBUG("writing uint64_t with value %" PRIu64 "\n", *v);
		m_json_writer.Uint64(*v);
	}
	virtual void on(float* v)
	{
		SG_SDEBUG("writing float with value %f\n", *v);
		m_json_writer.Double(*v);
	}
	virtual void on(float64_t* v)
	{
		SG_SDEBUG("writing double with value %f\n", *v);
		m_json_writer.Double(*v);
	}
	virtual void on(floatmax_t* v)
	{
		SG_SDEBUG("writing floatmax_t with value %f\n", *v);
		m_json_writer.Double(*v);
	}
	virtual void on(CSGObject** v)
	{
		if (*v)
		{
			SG_SDEBUG("writing SGObject\n");
			write_object(m_json_writer, this, wrap<CSGObject>(*v));
		}
		else
		{
			// nullptr
			m_json_writer.Null();
		}
	}
	virtual void on(SGVector<int32_t>* v)
	{
		SG_SDEBUG("writing SGVector<int>\n");
		write_vector(m_json_writer, &Writer::Int, v);
	}
	virtual void on(SGVector<float32_t>* v)
	{
		SG_SDEBUG("writing SGVector<float32_t>\n");
		write_vector(m_json_writer, &Writer::Double, v);
	}
	virtual void on(SGVector<float64_t>* v)
	{
		SG_SDEBUG("writing SGVector<float64_t>\n");
		write_vector(m_json_writer, &Writer::Double, v);
	}
	virtual void on(SGMatrix<int>* v)
	{
		SG_SDEBUG("writing SGMatrix<int>\n")
		write_matrix(m_json_writer, &Writer::Int, v);
	}
	virtual void on(SGMatrix<float32_t>* v)
	{
		SG_SDEBUG("writing SGMatrix<float32_t>\n");
		write_matrix(m_json_writer, &Writer::Double, v);
	}

	virtual void on(SGMatrix<float64_t>* v)
	{
		SG_SDEBUG("writing SGMatrix<float64_t>\n");
		write_matrix(m_json_writer, &Writer::Double, v);
	}

	virtual void on(vector<CSGObject*>* obj_array)
	{
		SG_SDEBUG("writing vector<CSGObject*>\n");
		m_json_writer.StartArray();
		for (auto v: *obj_array)
			on(&v);
		m_json_writer.EndArray();
	}

private:
	Writer& m_json_writer;
	SG_DELETE_COPY_AND_ASSIGN(JSONWriterVisitor);
};

template<typename Writer>
void write_object(Writer& writer, JSONWriterVisitor<Writer>* visitor, Some<CSGObject> object)
{
	writer.StartObject();
	writer.Key(kNameKey);
	writer.String(object->get_name());
	writer.Key(kGenericKey);
	writer.Int(object->get_generic());
	auto params = object->get_params();

	writer.Key(kParametersKey);
	writer.StartObject();
	for (const auto& p: params)
	{
		if (p.second->get_value().visitable())
		{
			writer.Key(p.first.c_str());
			p.second->get_value().visit(visitor);
		}
	}
	writer.EndObject();

	writer.EndObject();
}

using JsonWriter = Writer<COutputStreamAdapter>;

CJsonSerializer::CJsonSerializer() : CSerializer()
{
}

CJsonSerializer::~CJsonSerializer()
{
}

void CJsonSerializer::write(Some<CSGObject> object)
{
	COutputStreamAdapter adapter{ .m_stream = stream() };
	JsonWriter writer(adapter);
	auto writer_visitor =
		make_unique<JSONWriterVisitor<JsonWriter>>(writer);
	write_object(writer, writer_visitor.get(), object);
}
