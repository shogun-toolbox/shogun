#include <gtest/gtest.h>

#include <algorithm>

#include <shogun/io/ShogunErrc.h>
#include <shogun/io/serialization/BitserySerializer.h>
#include <shogun/io/serialization/BitseryDeserializer.h>

#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/serialization/JsonSerializer.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>

using namespace shogun;
using namespace shogun::io;
using namespace std;

class DummyOutputStream : public OutputStream
{
public:
	DummyOutputStream() : OutputStream(), m_buffer() {}

	error_condition close() override { return {}; }
	error_condition flush() override { return {}; }

	error_condition write(const void* buffer, int64_t size) override
	{
		copy(
			(char*)buffer,
			(char*)buffer + size,
			std::back_inserter(m_buffer));
		return {};
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

class DummyInputStream : public InputStream
{
public:
	DummyInputStream(const string& buffer) : InputStream(), m_buffer(buffer)
	{
		m_current_pos = m_buffer.cbegin();
	}

	error_condition read(string* buffer, int64_t size) override
	{
		require(buffer != nullptr, "Provided buffer should not be nullptr!");
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

template <typename T>
class SerializationTest : public ::testing::Test {};

using SerializerTypes = ::testing::Types<
	pair<JsonSerializer, JsonDeserializer>,
	pair<BitserySerializer, BitseryDeserializer>>;
TYPED_TEST_CASE(SerializationTest, SerializerTypes);

TYPED_TEST(SerializationTest, serialize)
{
	SGMatrix<float64_t> data {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
	auto df = std::make_shared<DenseFeatures<float64_t>>(data);
	auto obj = std::make_shared<GaussianKernel>(df, df, 2.0);

	auto serializer = std::make_shared<typename TypeParam::first_type>();
	auto stream = std::make_shared<DummyOutputStream>();
	serializer->attach(stream);
	serializer->write(obj);

	auto deserializer = std::make_shared<typename TypeParam::second_type>();
	auto istream = std::make_shared<DummyInputStream>(stream->buffer());
	deserializer->attach(istream);
	auto deser_obj = deserializer->read_object();

	ASSERT_TRUE(obj->equals(deser_obj));
}
