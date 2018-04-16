#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/serialization/JsonSerializer.h>

#include <algorithm>
#include <iostream>

#include <gtest/gtest.h>

using namespace shogun;

class CDummyOutputStream : public COutputStream
{
public:
	CDummyOutputStream() : COutputStream(), m_buffer()
	{
	}
	void write(void* buffer, size_t size)
	{
		std::copy(
		    (char*)buffer, (char*)buffer + size, std::back_inserter(m_buffer));
	}
	virtual const char* get_name() const
	{
		return "DummyOutputStream";
	}

private:
	std::vector<char> m_buffer;
};

TEST(JsonSerialization, basic_serializer)
{
	auto serializer = some<CJsonSerializer>();
	auto stream = some<CDummyOutputStream>();
	serializer->attach(stream);
	serializer->write(serializer);
}
