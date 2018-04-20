/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Viktor Gal
 */

#include <memory>

#include <shogun/base/class_list.h>
#include <shogun/base/macros.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

#include <rapidjson/reader.h>

using namespace shogun;

struct SGHandler : public rapidjson::BaseReaderHandler<UTF8<>, MyHandler> {
    bool Null() { cout << "Null()" << endl; return true; }
    bool Bool(bool b) { cout << "Bool(" << boolalpha << b << ")" << endl; return true; }
    bool Int(int i) { cout << "Int(" << i << ")" << endl; return true; }
    bool Uint(unsigned u) { cout << "Uint(" << u << ")" << endl; return true; }
    bool Int64(int64_t i) { cout << "Int64(" << i << ")" << endl; return true; }
    bool Uint64(uint64_t u) { cout << "Uint64(" << u << ")" << endl; return true; }
    bool Double(double d) { cout << "Double(" << d << ")" << endl; return true; }
    bool String(const char* str, SizeType length, bool copy) {
        cout << "String(" << str << ", " << length << ", " << boolalpha << copy << ")" << endl;
        return true;
    }
    bool StartObject() { cout << "StartObject()" << endl; return true; }
    bool Key(const char* str, SizeType length, bool copy) {
        cout << "Key(" << str << ", " << length << ", " << boolalpha << copy << ")" << endl;
        return true;
    }
    bool EndObject(SizeType memberCount) { cout << "EndObject(" << memberCount << ")" << endl; return true; }
    bool StartArray() { cout << "StartArray()" << endl; return true; }
    bool EndArray(SizeType elementCount) { cout << "EndArray(" << elementCount << ")" << endl; return true; }
};


template<typename RapidJsonReader>
class JSONReaderVisitor : public AnyVisitor
{
public:
	JSONReaderVisitor(RapidJsonReader& jr, rapidjson::Document* document):
		AnyVisitor(), m_json_reader(jr), m_document(document) {}

	virtual void on(bool* v)
	{
		SG_SDEBUG("reading bool")
		*v = m_json_reader.GetBool();
		SG_SDEBUG("%d\n", *v)
	}
	virtual void on(int32_t* v)
	{
		SG_SDEBUG("reading int32_t")
		*v = m_json_reader.GetInt();
		SG_SDEBUG("%d\n", *v)
	}
	virtual void on(int64_t* v)
	{
		SG_SDEBUG("reading int64_t")
		*v = m_json_reader.GetInt64();
		SG_SDEBUG("%d\n", *v)
	}
	virtual void on(float* v)
	{
		SG_SDEBUG("reading float: ")
		*v = (float32_t)m_json_reader.GetDouble();
		SG_SDEBUG("%f\n", *v)
	}
	virtual void on(double* v)
	{
		SG_SDEBUG("reading double: ")
		*v = m_json_reader.GetDouble();
		SG_SDEBUG("%f\n", *v)
	}
	virtual void on(CSGObject** v)
	{
		SG_SDEBUG("reading SGObject: ")
		*v = m_deser->read().get();
		/*
		std::string object_name;
		EPrimitiveType primitive_type;
		m_archive(object_name, primitive_type);
		SG_SDEBUG("%s %d\n", object_name.c_str(), primitive_type)
		if (*v == nullptr)
			SG_UNREF(*v);
		*v = create(object_name.c_str(), primitive_type);
		m_archive(**v);
		*/
	}
	virtual void on(SGVector<int>* v)
	{
		SG_SDEBUG("reading SGVector<int>\n")
	}
	virtual void on(SGVector<float>* v)
	{
		SG_SDEBUG("reading SGVector<float>\n")
	}
	virtual void on(SGVector<double>* v)
	{
		SG_SDEBUG("reading SGVector<double>\n")
	}
	virtual void on(SGMatrix<int>* v)
	{
		SG_SDEBUG("reading SGMatrix<int>>\n")
	}
	virtual void on(SGMatrix<float>* v)
	{
		SG_SDEBUG("reading SGMatrix<float>>\n")
	}
	virtual void on(SGMatrix<double>* v)
	{
		SG_SDEBUG("reading SGMatrix<double>>\n")
	}

private:
	RapidJsonReader& m_json_reader;
	rapidjson::Document* m_document;
};

class CIStreamAdapter
{
public:
	typedef char Ch;

	CIStreamAdapter(CInputStream* is): m_stream(is) {}

	Ch Peek() const
	{
		//int c = m_stream.peek();
	//	return c == std::char_traits<char>::eof() ? '\0' : (Ch)c;
	}

	Ch Take()
	{
	//	int c = m_stream.get();
	//	return c == std::char_traits<char>::eof() ? '\0' : (Ch)c;
	}

	size_t Tell() const
	{
	//	return (size_t)m_stream.tellg();
	}

	Ch* PutBegin() { assert(false); return 0; }
	void Put(Ch) { assert(false); }
	void Flush() { assert(false); }
	size_t PutEnd(Ch*) { assert(false); return 0; }

private:
	CInputStream* m_stream;
	SG_DELETE_COPY_AND_ASSIGN(CIStreamAdapter);
};


template<typename Reader>
Some<CSGObject> object_reader(Reader& reader)
{
	auto reader_visitor = std::make_unique<JSONReaderVisitor<rapidjson::Document>>(obj_json);
	if (!obj_json.IsObject())
		throw ShogunException("JSON value is not an object!");

	std::string obj_name(obj_json["name"].GetString());
	EPrimitiveType primitive_type((EPrimitiveType) obj_json["generic"].GetInt());
	auto obj = create(obj_name.c_str(), primitive_type);
	for (auto it = obj_json.MemberBegin(); it != obj_json.MemberEnd(); ++it)
	{
		auto param_name = it->name.GetString();
		if (!has(param_name))
			throw ShogunException(
				"cannot deserialize the object from file!");

		BaseTag tag(param_name);
		auto parameter = obj->get_parameter(tag);
		parameter.get_value().visit(reader_visitor.get());
		obj->update_parameter(tag, parameter.get_value());
	}
	return wrap<CSGObject>(obj);
}


CJsonDeserializer::CJsonDeserializer() : CDeserializer()
{
}

CJsonDeserializer::~CJsonDeserializer()
{
}

Some<CSGObject> CJsonDeserializer::read()
{
	CIStreamAdapter is(stream().get());
	rapidjson::Document obj_json;
	obj_json.ParseStream(is);
	auto reader_visitor = std::make_unique<JSONReaderVisitor<rapidjson::Document>>(obj_json, &obj_json);

	if (!obj_json.IsObject())
		throw ShogunException("JSON value is not an object!");

	std::string obj_name(obj_json["name"].GetString());
	EPrimitiveType primitive_type((EPrimitiveType) obj_json["generic"].GetInt());
	auto obj = create(obj_name.c_str(), primitive_type);
	for (auto it = obj_json.MemberBegin(); it != obj_json.MemberEnd(); ++it)
	{
		auto param_name = it->name.GetString();
		if (!has(param_name))
			throw ShogunException(
				"cannot deserialize the object from file!");

		BaseTag tag(param_name);
		auto parameter = obj->get_parameter(tag);
		parameter.get_value().visit(reader_visitor.get());
		obj->update_parameter(tag, parameter.get_value());
	}
	return wrap<CSGObject>(obj);
}
