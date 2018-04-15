/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Viktor Gal
 */

#include <memory>

#include <shogun/base/class_list.h>
#include <shogun/base/macros.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/ShogunErrc.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/util/converters.h>

#include <rapidjson/reader.h>
#include <rapidjson/document.h>

#include <iostream>

using namespace rapidjson;
using namespace shogun;
using namespace shogun::io;
using namespace std;

template<class ValueType>
class JSONReaderVisitor;

template<typename V>
CSGObject* object_reader(V& v, JSONReaderVisitor<V>* visitor);

extern const char* const kNameKey;
extern const char* const kGenericKey;
extern const char* const kParametersKey;

template<typename Reader, typename Fn, typename V>
void read_vector(Reader* reader, Fn f, V vector)
{
	auto json_array = reader->GetArray();
	if (!json_array.Size())
		return;
	vector->resize_vector(json_array.Size());
	auto it = json_array.Begin();
	index_t idx = 0;
	for (;it != json_array.End(); ++it, ++idx)
	{
		vector->vector[idx] = ((*it).*f)();
	}
}

template<typename Reader, typename Fn, typename V>
void read_matrix(Reader* reader, Fn f, V* matrix)
{
	auto json_array = reader->GetArray();
	if (!json_array.Size())
		return;

	SGVector<typename V::Scalar> v;
	auto it = json_array.Begin();
	read_vector(it, f, &v);
	V m(v.vlen, json_array.Size());
	index_t col = 0;
	// TODO: could do this with less memory consumption
	// i.e. directly set elements and not to do copies
	m.set_column(col, v);
	for (++col, ++it; it != json_array.End(); ++it, ++col)
	{
		read_vector(it, f, &v);
		m.set_column(col, v);
	}
	*matrix = m;
}

template<class ValueType>
class JSONReaderVisitor: public AnyVisitor
{
public:
	JSONReaderVisitor(): AnyVisitor() {}

	virtual void on(bool* v)
	{
		*v = m_current_value->GetBool();
	}
	virtual void on(int32_t* v)
	{
		*v = m_current_value->GetInt();
	}
	virtual void on(int64_t* v)
	{
		*v = m_current_value->GetInt64();
	}
	virtual void on(uint64_t* v)
	{
		*v = m_current_value->GetUint64();
	}
	virtual void on(float32_t* v)
	{
		*v = utils::safe_convert<float32_t>(m_current_value->GetDouble());
	}
	virtual void on(float64_t* v)
	{
		*v = m_current_value->GetDouble();
	}
	virtual void on(floatmax_t* v)
	{
		*v = utils::safe_convert<floatmax_t>(m_current_value->GetDouble());
	}
	virtual void on(CSGObject** v)
	{
		SG_SDEBUG("reading SGObject: ");
		*v = object_reader(m_current_value, this);
	}
	virtual void on(SGVector<int32_t>* v)
	{
		SG_SDEBUG("reading SGVector<int>: ")
		read_vector(m_current_value, &ValueType::GetInt, v);
	}
	virtual void on(SGVector<float32_t>* v)
	{
		SG_SDEBUG("reading SGVector<float32_t>: ")
		// FIXME: safe_convert should be checked!
		read_vector(m_current_value, &ValueType::GetDouble, v);
	}
	virtual void on(SGVector<float64_t>* v)
	{
		SG_SDEBUG("reading SGVector<float64_t>: ")
		read_vector(m_current_value, &ValueType::GetDouble, v);
	}
	virtual void on(SGMatrix<int32_t>* v)
	{
		SG_SDEBUG("reading SGMatrix<int>>: ")
		read_matrix(m_current_value, &ValueType::GetInt, v);
	}
	virtual void on(SGMatrix<float32_t>* v)
	{
		SG_SDEBUG("reading SGMatrix<float32_t>>: ")
		// FIXME: safe_convert should be checked!
		read_matrix(m_current_value, &ValueType::GetDouble, v);
	}
	virtual void on(SGMatrix<float64_t>* v)
	{
		SG_SDEBUG("reading SGMatrix<float64_t>>: ")
		read_matrix(m_current_value, &ValueType::GetDouble, v);
	}

	virtual void on(std::vector<CSGObject*>* v)
	{
		SG_SDEBUG("reading std::vector<CSGObject*>: ");
		for (auto& o: m_current_value->GetArray())
		{
			REQUIRE(o.IsObject(), "Vector of CSGObject should contain objects!")
			CSGObject* sg_obj = nullptr;
			on(&sg_obj);
			if (sg_obj != nullptr)
				v->push_back(sg_obj);
		}
	}

	void set(const ValueType* v)
	{
		m_current_value = v;
	}
private:
	const ValueType* m_current_value;
};

class CIStreamAdapter
{
public:
	typedef char Ch;

	CIStreamAdapter(Some<CInputStream> is): m_stream(is) {}

	Ch Peek() const
	{
		string c;
		auto current_pos = m_stream->tell();
		auto r = m_stream->read(&c, 1);
		if (!r)
		{
			m_stream->reset();
			m_stream->skip(current_pos);
			return c[0];
		}
		else if (io::is_out_of_range(r))
		{
			return '\0';
		}
		else
		{
			throw io::to_system_error(r);
		}
	}

	Ch Take()
	{
		string c;
		auto r = m_stream->read(&c, 1);
		if (r)
		{
			// eof
			if (io::is_out_of_range(r))
				return '\0';
			throw io::to_system_error(r);
		}
		return c[0];
	}

	size_t Tell() const
	{
		return utils::safe_convert<size_t>(m_stream->tell());
	}

	Ch* PutBegin() { assert(false); return 0; }
	void Put(Ch) { assert(false); }
	void Flush() { assert(false); }
	size_t PutEnd(Ch*) { assert(false); return 0; }

private:
	Some<CInputStream> m_stream;
	SG_DELETE_COPY_AND_ASSIGN(CIStreamAdapter);
};

template<typename V>
CSGObject* object_reader(const V* v, JSONReaderVisitor<V>* visitor)
{
	const V& value = *v;
	REQUIRE(v != nullptr, "Value should be set!");

	if (value.IsNull())
		return nullptr;

	REQUIRE(value.HasMember(kNameKey), "Not a valid serialized SGObject, it does not have a 'name'!")
	string obj_name(value[kNameKey].GetString());
	REQUIRE(value.HasMember(kGenericKey), "Not a valid serialized SGObject, it does not have a 'generic'!")
	EPrimitiveType primitive_type((EPrimitiveType) value[kGenericKey].GetInt());
	auto obj = create(obj_name.c_str(), primitive_type);
	REQUIRE(obj != nullptr, "Could not create '%s' class", obj_name.c_str())
	REQUIRE(value.HasMember(kParametersKey), "Not a valid serialized SGObject, it does not have 'parameters!")
	REQUIRE(value[kParametersKey].IsObject(), "Not a valid serialized SGObject!")
	auto obj_params = value[kParametersKey].GetObject();
	for_each(obj_params.MemberBegin(), obj_params.MemberEnd(),
		[&obj, &visitor](const auto& member) {
			visitor->set(&(member.value));
			obj->visit_parameter(BaseTag(member.name.GetString()), visitor);
		});
	return obj;
}

CJsonDeserializer::CJsonDeserializer() : CDeserializer()
{
}

CJsonDeserializer::~CJsonDeserializer()
{
}

Some<CSGObject> CJsonDeserializer::read()
{
	CIStreamAdapter is(stream());
	// FIXME: use SAX parser interface!
	Document reader;
	reader.ParseStream(is);
	auto reader_visitor =
		make_unique<JSONReaderVisitor<Document::ValueType>>();
	return wrap<CSGObject>(object_reader(
		dynamic_cast<Document::ValueType*>(&reader),
		reader_visitor.get())
	);
}
