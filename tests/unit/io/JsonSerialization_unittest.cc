#include <gtest/gtest.h>

#include <shogun/io/ShogunErrc.h>

#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/serialization/JsonSerializer.h>

#include <algorithm>

#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>

using namespace shogun;
using namespace shogun::io;
using namespace std;

class CDummyOutputStream : public COutputStream
{
public:
	CDummyOutputStream() : COutputStream(), m_buffer() {}

	error_condition close() override { return {}; }
	error_condition flush() override { return {}; }

	error_condition write(const void* buffer, int64_t size) override
	{
		copy(
			(char*)buffer,
			(char*)buffer + size,
			std::back_inserter(m_buffer));
	}
	const char* get_name() const override
	{
		return "DummyOutputStream";
	}

	string buffer() const
	{
		return m_buffer;
	}

private:
	string m_buffer;
};

class CDummyInputStream : public CInputStream
{
public:
	CDummyInputStream(const string& buffer) : CInputStream(), m_buffer(buffer)
	{
		m_current_pos = m_buffer.cbegin();
	}

	error_condition read(string* buffer, int64_t size) override
	{
		REQUIRE(buffer != nullptr, "Provided buffer should not be nullptr!");
		auto d = distance(m_current_pos, m_buffer.cend());
		if (d == 0)
		{
			return make_error_condition(ShogunErrc::OutOfRange);
		}

		if (size <= d)
		{
			buffer->resize(size);
			copy(m_current_pos, m_current_pos+size, buffer->begin());
			m_current_pos += size;
			return {};
		}
		else
		{
			buffer->resize(d);
			copy(m_current_pos, m_current_pos+d, buffer->begin());
			m_current_pos += d;
			return make_error_condition(ShogunErrc::OutOfRange);
		}
	}

	error_condition skip(int64_t bytes) override
	{
		if ((m_current_pos + bytes) != m_buffer.cend())
		{
			m_current_pos += bytes;
			return {};
		}
		else
		{
			m_current_pos = m_buffer.cend();
			return make_error_condition(ShogunErrc::OutOfRange);
		}
	}
	char peek() const
	{
		return m_current_pos == m_buffer.cend()
			? char_traits<char>::eof()
			: *m_current_pos;
	}

	int64_t tell() const override
	{
		return distance(m_buffer.cbegin(), m_current_pos);
	}

	void reset() override
	{
		m_current_pos = m_buffer.cbegin();
	}

	const char* get_name() const override
	{
		return "DummyInputStream";
	}

private:
	string m_buffer;
	string::const_iterator m_current_pos;
};


TEST(JsonSerialization, basic_serializer)
{
	SGMatrix<float64_t> data {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
	auto df = new CDenseFeatures<float64_t>(data);
	auto obj = some<CGaussianKernel>(df, df, 2.0);

	auto serializer = some<CJsonSerializer>();
	auto stream = some<CDummyOutputStream>();
	serializer->attach(stream);
	serializer->write(obj);

	auto deserializer = some<CJsonDeserializer>();
	auto istream = some<CDummyInputStream>(stream->buffer());
	deserializer->attach(istream);
	auto deser_obj = deserializer->read();

	ASSERT_TRUE(obj->equals(deser_obj));
}
