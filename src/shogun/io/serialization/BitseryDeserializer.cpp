/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <shogun/io/serialization/BitseryDeserializer.h>
#include <shogun/io/serialization/BitseryVisitor.h>
#include <shogun/io/ShogunErrc.h>
#include <shogun/util/converters.h>
#include <shogun/base/class_list.h>
#include <shogun/util/system.h>

#include <bitsery/bitsery.h>
#include <bitsery/traits/string.h>

using namespace bitsery;
using namespace shogun;
using namespace shogun::io;
using namespace std;

template<class S>
class BitseryReaderVisitor: public detail::BitseryVisitor<S, BitseryReaderVisitor<S>>
{
public:
	BitseryReaderVisitor(S& s):
		detail::BitseryVisitor<S,BitseryReaderVisitor<S>>(s) {}

	void on_complex(S& s, complex128_t* v)
	{
		float64_t real, imag;
		s.value8b(real);
		s.value8b(imag);
		v->real(real);
		v->imag(imag);
	}

	void on_floatmax(S& s, floatmax_t* v)
	{
		uint64_t msb, lsb;
		s.value8b(msb);
		s.value8b(lsb);

		// FIXME: check array[0] == array[1]
		uint64_t array[2];
		array[utils::is_big_endian() ? 1 : 0] = msb;
		array[utils::is_big_endian() ? 0 : 1] = lsb;

		*v = *reinterpret_cast<floatmax_t*>(array);
		SG_SDEBUG("read floatmax_t with value %Lf\n", *v);
	}

	void on_object(S& s, CSGObject** v)
	{
		SG_SDEBUG("reading SGObject: ");
		if (*v != nullptr)
			SG_UNREF(*v);
		*v = object_reader(s, this);
		if (*v != nullptr)
			SG_REF(*v);
	}

private:
	SG_DELETE_COPY_AND_ASSIGN(BitseryReaderVisitor);
};

struct InputStreamAdapter
{
	typedef char TValue;
	typedef void TIterator;

	void read(TValue* buffer, size_t bytes)
	{
		m_status = m_stream->read(&m_buffer, bytes);
		// FIXME: copying!
		copy_n(m_buffer.begin(), bytes, buffer);
	}

	ReaderError error() const
	{
		if (!m_status)
			return ReaderError::NoError;
		return io::is_out_of_range(m_status)
			? ReaderError::DataOverflow
			: ReaderError::ReadingError;
	}

	bool isCompletedSuccessfully() const
	{
		return io::is_out_of_range(m_status);
	}

	void setError(ReaderError error)
	{
		// ignore
	}

	Some<CInputStream> m_stream;
	string m_buffer;
	error_condition m_status;
};

template<typename Reader>
CSGObject* object_reader(Reader& reader, BitseryReaderVisitor<Reader>* visitor, CSGObject* _this = nullptr)
{
	size_t obj_magic;
	reader.value8b(obj_magic);
	if (obj_magic == detail::kNullObjectMagic)
		return nullptr;

	string obj_name;
	reader.text1b(obj_name, 64);
	uint16_t primitive_type;
	reader.value2b(primitive_type);
	CSGObject* obj = nullptr;
	if (_this)
	{
		REQUIRE(_this->get_name() == obj_name, "");
		REQUIRE(_this->get_generic() == static_cast<EPrimitiveType>(primitive_type), "");
		obj = _this;
	}
	else
	{
		obj = create(obj_name.c_str(), static_cast<EPrimitiveType>(primitive_type));
	}
	if (obj == nullptr)
		throw runtime_error("Trying to deserializer and unknown object!");

	size_t num_params;
	reader.value8b(num_params);
	for (size_t i = 0; i < num_params; ++i)
	{
		string param_name;
		reader.text1b(param_name, 64);
		obj->visit_parameter(BaseTag(param_name), visitor);
	}
	return obj;
}

using InputAdapter = AdapterReader<InputStreamAdapter, bitsery::DefaultConfig>;
using BitseryDeserializer = BasicDeserializer<InputAdapter>;

CBitseryDeserializer::CBitseryDeserializer() : CDeserializer()
{
}

CBitseryDeserializer::~CBitseryDeserializer()
{
}

Some<CSGObject> CBitseryDeserializer::read_object()
{
	InputStreamAdapter adapter{ .m_stream = stream() };
	BitseryDeserializer deser {std::move(adapter)};
	BitseryReaderVisitor<BitseryDeserializer> reader_visitor(deser);
	return wrap<CSGObject>(object_reader(deser, addressof(reader_visitor)));
}

void CBitseryDeserializer::read(CSGObject* _this)
{
	InputStreamAdapter adapter{ .m_stream = stream() };
	BitseryDeserializer deser {std::move(adapter)};
	BitseryReaderVisitor<BitseryDeserializer> reader_visitor(deser);
	object_reader(deser, addressof(reader_visitor), _this);
}
