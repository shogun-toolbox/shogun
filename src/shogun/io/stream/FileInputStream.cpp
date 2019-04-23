#include <shogun/io/stream/FileInputStream.h>

using namespace std;
using namespace shogun::io;

FileInputStream::FileInputStream(RandomAccessFile* src, bool free):
	InputStream(), m_src(src), m_free(free), m_pos(0) {}
FileInputStream::~FileInputStream()
{
	if (m_free)
		delete m_src;
}

error_condition FileInputStream::read(string* buffer, int64_t size)
{
	if (size < 0)
		return make_error_condition(errc::invalid_argument);

	buffer->clear();
	buffer->resize(size);
	char* result_buffer = &(*buffer)[0];
	string_view data;
	auto r = m_src->read(m_pos, size, &data, result_buffer);

	if (data.data() != result_buffer)
		memmove(result_buffer, data.data(), data.size());

	buffer->resize(data.size());
	if (!r || io::is_out_of_range(r))
		m_pos += data.size();
	return r;
}

static constexpr int64_t kMaxSkipSize = 8 * 1024 * 1024;

error_condition FileInputStream::skip(int64_t bytes)
{
	if (bytes < 0)
		return make_error_condition(errc::invalid_argument);

	unique_ptr<char[]> scratch(new char[kMaxSkipSize]);
	if (bytes > 0)
	{
		// try quickly jumping on the position
		string_view data;
		auto r = m_src->read(m_pos + bytes - 1, 1, &data, scratch.get());
		if ((!r || io::is_out_of_range(r)) && data.size() == 1)
		{
			m_pos += bytes;
			return {};
		}
	}

	while (bytes > 0)
	{
		// skip until we can and then
		auto bytes_to_read = min<int64_t>(kMaxSkipSize, bytes);
		string_view data;
		auto r = m_src->read(m_pos, bytes, &data, scratch.get());
		if (!r || io::is_out_of_range(r))
			m_pos += data.size();
		else
		 	return r;

		if (data.size() < bytes_to_read)
		  return make_error_condition(ShogunErrc::OutOfRange);
		bytes -= bytes_to_read;
	}
	return {};
}

int64_t FileInputStream::tell() const
{
	return m_pos;
}

void FileInputStream::reset()
{
	m_pos = 0;
}
