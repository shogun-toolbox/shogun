/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <shogun/io/serialization/BitserySerializer.h>
#include <shogun/io/serialization/BitseryVisitor.h>
#include <shogun/io/ShogunErrc.h>
#include <shogun/util/converters.h>

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

	void on_object(Writer& writer, CSGObject** v)
	{
		if (*v)
		{
			SG_SDEBUG("writing SGObject\n");
			SG_REF(*v);
			write_object(writer, this, wrap<CSGObject>(*v));
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

	Some<COutputStream> m_stream;
	size_t written_bytes = 0;
};

// cannot use context because of circular dependency :(
template<typename Writer>
void write_object(Writer& writer, BitseryWriterVisitor<Writer>* visitor, Some<CSGObject> o)
{
	writer.value8b(sizeof(o.get()));
	string name(o->get_name());
	writer.text1b(name, 64);
	writer.value2b(static_cast<uint16_t>(o->get_generic()));
	auto params = o->get_params();
	for (auto it = params.begin(); it != params.end();)
	{
		if (!it->second->get_value().visitable())
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
}

using OutputAdapter = AdapterWriter<OutputStreamAdapter, bitsery::DefaultConfig>;
using BitserySerializer = BasicSerializer<OutputAdapter>;

CBitserySerializer::CBitserySerializer() : CSerializer()
{
}

CBitserySerializer::~CBitserySerializer()
{
}

void CBitserySerializer::write(Some<CSGObject> object)
{
	OutputStreamAdapter adapter{ .m_stream = stream() };
 	BitserySerializer serializer {std::move(adapter)};
 	BitseryWriterVisitor<BitserySerializer> writer_visitor(serializer);
 	write_object(serializer, addressof(writer_visitor), object);
}
