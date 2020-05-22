/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Viktor Gal
 */

#include <memory>
#include <stack>
#include <utility>

#include <shogun/base/class_list.h>
#include <shogun/base/macros.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/ShogunErrc.h>
#include <shogun/util/converters.h>
#include <shogun/util/system.h>

#include <rapidjson/reader.h>
#include <rapidjson/document.h>

using namespace rapidjson;
using namespace shogun;
using namespace shogun::io;
using namespace std;

template<class ValueType>
class JSONReaderVisitor;

template<typename V>
std::shared_ptr<SGObject> object_reader(V& v, JSONReaderVisitor<V>* visitor);

extern const char* const kNameKey;
extern const char* const kGenericKey;
extern const char* const kParametersKey;

template<class ValueType>
class JSONReaderVisitor: public AnyVisitor
{
	using ReverseConstIterator = reverse_iterator<typename ValueType::ConstValueIterator>;

public:
	JSONReaderVisitor(): AnyVisitor() {}
	~JSONReaderVisitor() override {}

	void on(bool* v) override
	{
		*v = next_element<bool>(&ValueType::GetBool);
		SG_DEBUG("read bool with value {}", *v);
	}
	void on(std::vector<bool>::reference* v) override
	{
		*v = next_element<bool>(&ValueType::GetBool);
		SG_DEBUG("read bool with value {}", *v);
	}
	void on(char* v) override
	{
		*v = utils::safe_convert<char>(next_element<int32_t>(&ValueType::GetInt));
		SG_DEBUG("read char with value {}", *v);
	}
	void on(int8_t* v) override
	{
		*v = utils::safe_convert<int8_t>(next_element<int32_t>(&ValueType::GetInt));
		SG_DEBUG("read int8_t with value {}", *v);
	}
	void on(uint8_t* v) override
	{
		*v = utils::safe_convert<uint8_t>(next_element<uint32_t>(&ValueType::GetUint));
		SG_DEBUG("read uint8_t with value {}", *v);
	}
	void on(int16_t* v) override
	{
		*v = utils::safe_convert<int16_t>(next_element<int32_t>(&ValueType::GetInt));
		SG_DEBUG("read int16_t with value {}", *v);
	}
	void on(uint16_t* v) override
	{
		*v = utils::safe_convert<uint16_t>(next_element<uint32_t>(&ValueType::GetUint));
		SG_DEBUG("read uint16_t with value {}", *v);
	}
	void on(int32_t* v) override
	{
		*v = next_element<int32_t>(&ValueType::GetInt);
		SG_DEBUG("read int32_t with value {}", *v);
	}
	void on(uint32_t* v) override
	{
		*v = next_element<uint32_t>(&ValueType::GetUint);
		SG_DEBUG("read uint32_t with value {}", *v);
	}
	void on(int64_t* v) override
	{
		*v = next_element<int64_t>(&ValueType::GetInt64);
		SG_DEBUG("read int64_t with value {}", *v);
	}
	void on(uint64_t* v) override
	{
		*v = next_element<uint64_t>(&ValueType::GetUint64);
		SG_DEBUG("read uint64_t with value {}", *v);
	}
	void on(float32_t* v) override
	{
		*v = utils::safe_convert<float32_t>(next_element<float64_t>(&ValueType::GetDouble));
		SG_DEBUG("read float with value {}", *v);
	}
	void on(float64_t* v) override
	{
		*v = next_element<float64_t>(&ValueType::GetDouble);
		SG_DEBUG("read double with value {}", *v);
	}
	void on(floatmax_t* v) override
	{
		assert(!m_value_stack.empty());
		auto _v = m_value_stack.top();
		auto floatmax_pair = _v->GetArray();
		assert(floatmax_pair.Size() == 2);
		uint64_t array[2];
		// FIXME: check array[0] == array[1]
		array[utils::is_big_endian() ? 1 : 0] = floatmax_pair[0].GetUint64();
		array[utils::is_big_endian() ? 0 : 1] = floatmax_pair[1].GetUint64();
		m_value_stack.pop();

		*v = *reinterpret_cast<floatmax_t*>(array);
		SG_DEBUG("read floatmax_t with value {}", *v);
	}
	void on(complex128_t* v) override
	{
		assert(!m_value_stack.empty());
		auto _v = m_value_stack.top();
		auto complex_pair = _v->GetArray();
		assert(complex_pair.Size() == 2);
		v->real(complex_pair[0].GetDouble());
		v->imag(complex_pair[1].GetDouble());
		m_value_stack.pop();
	}
	void on(std::string* v) override
	{
		*v = next_element<std::string>(&ValueType::GetString);
		SG_DEBUG("reading std::string: {}", v->c_str());
	}
	void on(AutoValueEmpty* v) override
	{
		const auto& temp = next_element<std::string>(&ValueType::GetString);
		SG_DEBUG("read empty auto value: {}", temp);
	}
	void on(std::shared_ptr<SGObject>* v) override
	{
		SG_DEBUG("reading SGObject: ");
		*v = object_reader(m_value_stack.top(), this);
		m_value_stack.pop();
	}
	void enter_auto_value(bool* is_empty) override 
	{
		*is_empty = m_value_stack.top()->IsString();
	}
	void enter_matrix(index_t* rows, index_t* cols) override
	{
		auto json_array = m_value_stack.top()->GetArray();
		m_value_stack.pop();
		*cols = json_array.Size();
		if (*cols != 0)
		{
			ReverseConstIterator col_begin(json_array.End());
			ReverseConstIterator col_end(json_array.Begin());
			*rows = col_begin->GetArray().Size();
			SG_DEBUG("reading matrix of size: {} x {}", *rows, *cols);
			do
			{
				auto json_row = col_begin->GetArray();
				ReverseConstIterator row_begin(json_row.End());
				ReverseConstIterator row_end(json_row.Begin());
				do
				{
					m_value_stack.emplace(addressof(*row_begin));
				} while (++row_begin != row_end);
			} while (++col_begin != col_end);
		}
	}
	void enter_vector(index_t* size) override
	{
		read_array(size, "SGVector");
	}
	void enter_std_vector(size_t* size) override
	{
		read_array(size, "std::vector");
	}
	void enter_map(size_t* size) override
	{
		auto json_array = m_value_stack.top()->GetArray();
		m_value_stack.pop();
		*size = utils::safe_convert<size_t>(json_array.Size());
		SG_DEBUG("reading map of size: {}", *size);
		if (*size == 0)
			return;
		for (auto it = json_array.Begin(); it != json_array.End(); ++it)
		{
			auto json_row = it->GetArray();
			for (auto row_it = json_row.Begin();
				row_it != json_row.End(); ++row_it)
			{
				m_value_stack.emplace(addressof(*row_it));
			}
		}
	}

	void push(const ValueType* v)
	{
		m_value_stack.emplace(v);
	}

	void enter_matrix_row(index_t *rows, index_t *cols) override {}
	void exit_matrix_row(index_t *rows, index_t *cols) override {}
	void exit_matrix(index_t* rows, index_t* cols) override {}
	void exit_vector(index_t* size) override {}
	void exit_std_vector(size_t* size) override {}
	void exit_map(size_t* size) override {}
private:
	template<typename T, typename Fn>
	T next_element(Fn f)
	{
		if (m_value_stack.empty())
			return 0;

		auto _v = m_value_stack.top();
		auto r = (_v->*f)();
		m_value_stack.pop();

		return r;
	}

	template<class T>
	void read_array(T* size, const std::string& type)
	{
		auto json_array = m_value_stack.top()->GetArray();
		m_value_stack.pop();
		*size = utils::safe_convert<T>(json_array.Size());
		SG_DEBUG("reading '{}' of size: {}", type.c_str(), *size);
		if (*size == 0)
			return;

		ReverseConstIterator rbegin(json_array.End());
		ReverseConstIterator rend(json_array.Begin());
		do
		{
			m_value_stack.emplace(addressof(*rbegin));
		} while (++rbegin != rend);
	}

private:
	stack<const ValueType*> m_value_stack;
	SG_DELETE_COPY_AND_ASSIGN(JSONReaderVisitor);
};

class IStreamAdapter
{
public:
	typedef char Ch;

	IStreamAdapter(std::shared_ptr<InputStream> is, size_t buffer_size = 65536):
		m_stream(std::move(is)),
		m_buffer_size(buffer_size)
	{
		m_buffer.reserve(m_buffer_size);
		read();
	}


	~IStreamAdapter()
	{
		m_buffer.clear();
	}

	Ch Peek() const
	{
		return m_buffer[m_pos];
	}

	Ch Take()
	{
		Ch c = m_buffer[m_pos];
		read();
		return c;
	}

	size_t Tell() const
	{
		return utils::safe_convert<size_t>(m_stream->tell()) - (m_limit + 1 - m_pos);
	}


	Ch* PutBegin() { assert(false); return 0; }
	void Put(Ch) { assert(false); }
	void Flush() { assert(false); }
	size_t PutEnd(Ch*) { assert(false); return 0; }
private:
	void read()
	{
		if (m_pos < m_limit)
		{
			++m_pos;
		}
		else if (!m_eof)
		{
			auto ec = m_stream->read(&m_buffer, m_buffer_size);
			m_pos = 0;
			m_limit = m_buffer.size() - 1;
			if (m_buffer.empty() || io::is_out_of_range(ec))
			{
				m_buffer.append("\0");
				++m_limit;
				m_eof = true;
			}
			else if (ec)
				throw io::to_system_error(ec);
		}
	}

private:
	std::shared_ptr<InputStream> m_stream;
	string m_buffer;
	size_t m_buffer_size;
	size_t m_pos = 0;
	size_t m_limit = 0;
	bool m_eof = false;
	SG_DELETE_COPY_AND_ASSIGN(IStreamAdapter);
};

template<typename V>
std::shared_ptr<SGObject> object_reader(const V* v, JSONReaderVisitor<V>* visitor, const std::shared_ptr<SGObject>& _this = nullptr)
{
	const V& value = *v;
	require(v != nullptr, "Value should be set!");

	if (value.IsNull())
		return nullptr;

	require(value.HasMember(kNameKey), "Not a valid serialized SGObject, it does not have a 'name'!");
	string obj_name(value[kNameKey].GetString());
	require(value.HasMember(kGenericKey), "Not a valid serialized SGObject, it does not have a 'generic'!");
	EPrimitiveType primitive_type((EPrimitiveType) value[kGenericKey].GetInt());
	std::shared_ptr<SGObject> obj = nullptr;
	if (_this)
	{
		require(_this->get_name() == obj_name, "");
		require(_this->get_generic() == primitive_type, "");
		obj = _this;
	}
	else
	{
		obj = create(obj_name.c_str(), primitive_type);
	}
	require(obj != nullptr, "Could not create '{}' class", obj_name.c_str());
	require(value.HasMember(kParametersKey), "Not a valid serialized SGObject, it does not have 'parameters!");
	require(value[kParametersKey].IsObject(), "Not a valid serialized SGObject!");
	auto obj_params = value[kParametersKey].GetObject();

	try
	{
		pre_deserialize(obj);

		for_each(obj_params.MemberBegin(), obj_params.MemberEnd(),
			[&obj, &visitor](const auto& member) {
				visitor->push(addressof(member.value));
				obj->visit_parameter(BaseTag(member.name.GetString()), visitor);
			});

		post_deserialize(obj);
	}
	catch(ShogunException& e)
	{
		io::warn("Error while deserializeing {}: ShogunException: "
			"{}", obj_name.c_str(), e.what());
		return nullptr;
	}

	return obj;
}

JsonDeserializer::JsonDeserializer() : Deserializer()
{
}

JsonDeserializer::~JsonDeserializer()
{
}

std::shared_ptr<SGObject> JsonDeserializer::read_object()
{
	IStreamAdapter is(stream());
	// FIXME: use SAX parser interface!
	Document reader;
	reader.ParseStream<kParseNanAndInfFlag>(is);
	auto reader_visitor =
		make_unique<JSONReaderVisitor<Document::ValueType>>();
	return object_reader(
		dynamic_cast<Document::ValueType*>(&reader),
		reader_visitor.get()
	);
}

void JsonDeserializer::read(std::shared_ptr<SGObject> _this)
{
	IStreamAdapter is(stream());
	// FIXME: use SAX parser interface!
	Document reader;
	reader.ParseStream<kParseNanAndInfFlag>(is);
	auto reader_visitor =
		make_unique<JSONReaderVisitor<Document::ValueType>>();
	object_reader(dynamic_cast<Document::ValueType*>(&reader),
		reader_visitor.get(), _this);
}
