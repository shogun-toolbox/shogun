/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <shogun/io/serialization/BitserySerializer.h>
#include <shogun/io/serialization/BitseryVisitor.h>
#include <shogun/io/ShogunErrc.h>
#include <shogun/util/converters.h>
#include <shogun/util/system.h>

#include <bitsery/bitsery.h>
#include <bitsery/traits/string.h>

using namespace bitsery;
using namespace shogun;
using namespace shogun::io;
using namespace std;

template<class Writer>
class BitseryWriterVisitor : public detail::BitseryVisitor<Writer, BitseryWriterVisitor<Writer>>
{
public:
	BitseryWriterVisitor(Writer& w):
		detail::BitseryVisitor<Writer,BitseryWriterVisitor<Writer>>(w) {}

	void on_complex(Writer& writer, complex128_t* v)
	{
		writer.value8b(v->real());
		writer.value8b(v->imag());
	}

	void on_floatmax(Writer& writer, floatmax_t* v)
	{
		SG_DEBUG("writing floatmax_t with value {}", *v);
		uint64_t msb, lsb;
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
		writer.value8b(msb);
		writer.value8b(lsb);
	}

	void on_object(Writer& writer, std::shared_ptr<SGObject>* v)
	{
		if (*v)
		{
			SG_DEBUG("writing SGObject: {}", (*v)->get_name());
			write_object(writer, this, *v);
		}
		else
		{
			// nullptr
			writer.value8b(detail::kNullObjectMagic);
		}
	}
};

struct OutputStreamAdapter
{
	typedef void TValue;

	void write(const TValue* buffer, size_t bytes)
	{
		auto ec = m_stream->write(buffer, bytes);
		if(ec)
			throw io::to_system_error(ec);
		written_bytes += bytes;
	}

	void flush()
	{
		m_stream->flush();
	}

	size_t writtenBytesCount() const
	{
		return written_bytes;
	}

	std::shared_ptr<OutputStream> m_stream;
	size_t written_bytes = 0;
};

// cannot use context because of circular dependency :(
template<typename Writer>
void write_object(Writer& writer, BitseryWriterVisitor<Writer>* visitor, std::shared_ptr<SGObject> o) noexcept(false)
{
	pre_serialize(o);
	writer.value8b(sizeof(o.get()));
	string name(o->get_name());
	writer.text1b(name, 64);
	writer.value2b(static_cast<uint16_t>(o->get_generic()));
	auto params = o->get_params();
	for (auto it = params.begin(); it != params.end();)
	{
		if (!it->second->get_value().visitable() || !it->second->get_value().cloneable())
			it = params.erase(it);
		else
			++it;
	}
	writer.value8b(params.size());
	for (const auto& p: params)
	{
		writer.text1b(p.first, 64);
		p.second->get_value().visit(visitor);
	}
	post_serialize(o);
}

using OutputAdapter = AdapterWriter<OutputStreamAdapter, bitsery::DefaultConfig>;
using BitserySer = BasicSerializer<OutputAdapter>;

BitserySerializer::BitserySerializer() : Serializer()
{
}

BitserySerializer::~BitserySerializer()
{
}

void BitserySerializer::write(std::shared_ptr<SGObject> object) noexcept(false)
{
	OutputStreamAdapter adapter { stream() };
 	BitserySer serializer {std::move(adapter)};
 	BitseryWriterVisitor<BitserySer> writer_visitor(serializer);
 	write_object(serializer, addressof(writer_visitor), object);
}
