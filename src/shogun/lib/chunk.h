#ifndef _CHUNK_H_
#define _CHUNK_H_

#include <string>

namespace shogun
{
class Chunk
{
public:
	using size_type = size_t;

	Chunk(): m_data(nullptr), m_size(0) {}
	Chunk(const char* d, size_t n) : m_data(d), m_size(n) {}
	Chunk(const std::string& s) : m_data(s.data()), m_size(s.size()) {}
	Chunk(const char* s) : m_data(s), m_size(strlen(s)) {}

	const char* data() const { return m_data; }
	size_t size() const { return m_size; }

	bool empty() const { return m_size == 0; }

	typedef const char* const_iterator;
	typedef const char* iterator;
	iterator begin() const { return m_data; }
	iterator end() const { return m_data + m_size; }

	static const size_t npos = size_type(-1);

	char operator[](size_t n) const
	{
		assert(n < size());
		return m_data[n];
	}

	 std::string to_string() const { return std::string(m_data, m_size); }

private:
	const char* m_data;
	size_t m_size;
};

}

#endif /** _CHUNK_H_ **/