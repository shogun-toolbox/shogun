/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Viktor Gal
 */

#include <memory>
#include <stack>

#include <shogun/io/ShogunErrc.h>
#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/util/converters.h>
#include <shogun/util/system.h>

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

struct OutputStreamAdapter
{
	typedef char Ch;
	void Put(Ch c)
	{
		if(auto ec = m_stream->write(&c, 1))
			throw io::to_system_error(ec);
	}

	void Flush()
	{
		m_stream->flush();
	}

	std::shared_ptr<OutputStream> m_stream;
};

template<typename Writer> void write_object(Writer& writer, std::shared_ptr<SGObject> object);

template<class Writer>
class JSONWriterVisitor : public AnyVisitor
{
	const int64_t in_object = -1;
public:
	JSONWriterVisitor(Writer& jw):
		AnyVisitor(), m_json_writer(jw) {}

	~JSONWriterVisitor() override {}

	void on(bool* v) override
	{
		SG_DEBUG("writing bool with value {}", *v);
		m_json_writer.Bool(*v);
		close_container();
	}
	void on(std::vector<bool>::reference* v) override
	{
		SG_DEBUG("writing bool with value {}", *v);
		m_json_writer.Bool(*v);
		close_container();
	}
	void on(char* v) override
	{
		SG_DEBUG("writing char with value {}", *v);
		m_json_writer.Int(*v);
		close_container();
	}
	void on(int8_t* v) override
	{
		SG_DEBUG("writing int8_t with value {}", *v);
		m_json_writer.Int(*v);
		close_container();
	}
	void on(uint8_t* v) override
	{
		SG_DEBUG("writing uint8_t with value {}", *v);
		m_json_writer.Uint(*v);
		close_container();
	}
	void on(int16_t* v) override
	{
		SG_DEBUG("writing int16_t with value {}", *v);
		m_json_writer.Int(*v);
		close_container();
	}
	void on(uint16_t* v) override
	{
		SG_DEBUG("writing uint16_t with value {}", *v);
		m_json_writer.Uint(*v);
		close_container();
	}
	void on(int32_t* v) override
	{
		SG_DEBUG("writing int32_t with value {}", *v);
		m_json_writer.Int(*v);
		close_container();
	}
	void on(uint32_t* v) override
	{
		SG_DEBUG("writing uint32_t with value {}", *v);
		m_json_writer.Uint(*v);
		close_container();
	}
	void on(int64_t* v) override
	{
		SG_DEBUG("writing int64_t with value {}", *v);
		m_json_writer.Int64(*v);
		close_container();
	}
	void on(uint64_t* v) override
	{
		SG_DEBUG("writing uint64_t with value {}", *v);
		m_json_writer.Uint64(*v);
		close_container();
	}
	void on(float* v) override
	{
		SG_DEBUG("writing float with value {}", *v);
		m_json_writer.Double(*v);
		close_container();
	}
	void on(float64_t* v) override
	{
		SG_DEBUG("writing double with value {}", *v);
		m_json_writer.Double(*v);
		close_container();
	}
	void on(floatmax_t* v) override
	{
		SG_DEBUG("writing floatmax_t with value {}", *v);
		uint64_t msb, lsb;
		m_json_writer.StartArray();
		uint64_t *array = reinterpret_cast<uint64_t*>(v);
		auto array_size = sizeof(floatmax_t)/sizeof(uint64_t);
		if (array_size == 2)
		{
			msb = utils::is_big_endian() ? array[1] : array[0];
			lsb = utils::is_big_endian() ? array[0] : array[1];
		}
		if (array_size < 2)
		{
			msb = array[0];
			lsb = array[0];
		}
		else
		{
			std::overflow_error("Could not represent floatmax_t with with 2 uint64_t!");
		}
		// write in little endian format
		m_json_writer.Uint64(msb);
		m_json_writer.Uint64(lsb);
		m_json_writer.EndArray();
		close_container();
	}
	void on(complex128_t* v) override
	{
		SG_DEBUG("writing complex128_t with value ({}, {})", v->real(), v->imag());
		m_json_writer.StartArray();
		m_json_writer.Double(v->real());
		m_json_writer.Double(v->imag());
		m_json_writer.EndArray();
		close_container();
	}
	void on(std::string* v) override
	{
		SG_DEBUG("writing std::string with value {}", v->c_str());
		m_json_writer.String(v->c_str());
	}
	void on(std::shared_ptr<SGObject>* v) override
	{
		if (*v)
		{
			SG_DEBUG("writing SGObject: {}", (*v)->get_name());
			write_object(m_json_writer, this, *v);
		}
		else
		{
			// nullptr
			m_json_writer.Null();
		}
		close_container();
	}
	void enter_matrix(index_t* rows, index_t* cols) override
	{
		SG_DEBUG("writing matrix of size: {} x {}", *rows, *cols);
		m_json_writer.StartArray();
		if (*cols == 0 || *rows == 0)
		{
			m_json_writer.EndArray();
		}
		else
		{
			m_remaining.emplace(*rows, *cols);
			m_remaining.emplace(*rows, 0LL);
			m_json_writer.StartArray();
		}
	}
	void enter_vector(index_t* size) override
	{
		SG_DEBUG("writing vector of size: {}", *size);
		m_json_writer.StartArray();
		if (*size == 0)
			m_json_writer.EndArray();
		else
			m_remaining.emplace(utils::safe_convert<int64_t>(*size), 0LL);
	}
	void enter_std_vector(size_t* size) override
	{
		SG_DEBUG("writing std::vector of size: {}", *size);
		m_json_writer.StartArray();
		if (*size == 0)
			m_json_writer.EndArray();
		else
			m_remaining.emplace(utils::safe_convert<int64_t>(*size), 0LL);
	}
	void enter_map(size_t* size) override
	{
		SG_DEBUG("writing map of size: {}", *size);
		m_json_writer.StartArray();
		if (*size == 0)
		{
			m_json_writer.EndArray();
		}
		else
		{
			m_remaining.emplace(utils::safe_convert<int64_t>(2), *size);
			m_remaining.emplace(utils::safe_convert<int64_t>(2), 0LL);
		}
	}

	void start_object()
	{
		m_remaining.emplace(in_object, 0LL);
	}
	void end_object()
	{
		assert(get<0>(m_remaining.top()) == in_object);
		m_remaining.pop();
	}

	void enter_matrix_row(index_t *rows, index_t *cols) override {}
	void exit_matrix_row(index_t *rows, index_t *cols) override {}
	void exit_matrix(index_t* rows, index_t* cols) override {}
	void exit_vector(index_t* size) override {}
	void exit_std_vector(size_t* size) override {}
	void exit_map(size_t* size) override {}
private:
	inline void close_container()
	{
		if (m_remaining.empty() || get<0>(m_remaining.top()) == in_object)
			return;

		auto& remaining = get<0>(m_remaining.top());
		if (remaining > 0 && --remaining == 0)
		{
			m_remaining.pop();
			m_json_writer.EndArray();

			auto& cols_remaining = get<1>(m_remaining.top());
			if (cols_remaining > 0)
			{
				if (--cols_remaining == 0)
				{
					m_remaining.pop();
					m_json_writer.EndArray();
				}
				else
				{
					m_remaining.emplace(get<0>(m_remaining.top()), 0LL);
					m_json_writer.StartArray();
				}
			}
			else
			{
					close_container();
			}
		}
	}
private:
	Writer& m_json_writer;
	stack<tuple<int64_t, int64_t>> m_remaining;
	SG_DELETE_COPY_AND_ASSIGN(JSONWriterVisitor);
};

template<typename Writer>
void write_object(Writer& writer, JSONWriterVisitor<Writer>* visitor, const std::shared_ptr<SGObject>& object) noexcept(false)
{
	pre_serialize(object);

	visitor->start_object();
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
		if (p.second->get_value().visitable() && p.second->get_value().cloneable())
		{
			writer.Key(p.first.c_str());
			p.second->get_value().visit(visitor);
		}
	}
	writer.EndObject();

	writer.EndObject();

	visitor->end_object();
	post_serialize(object);
}

using JsonWriter = Writer<OutputStreamAdapter, UTF8<>, UTF8<>, CrtAllocator, kWriteNanAndInfFlag>;

JsonSerializer::JsonSerializer() : Serializer()
{
}

JsonSerializer::~JsonSerializer()
{
}

void JsonSerializer::write(std::shared_ptr<SGObject> object) noexcept(false)
{
	OutputStreamAdapter adapter { stream() };
	JsonWriter writer(adapter);
	auto writer_visitor =
		make_unique<JSONWriterVisitor<JsonWriter>>(writer);
	write_object(writer, writer_visitor.get(), object);
}
