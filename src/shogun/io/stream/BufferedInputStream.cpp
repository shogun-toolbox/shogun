#include <shogun/io/stream/BufferedInputStream.h>
#include <shogun/io/ShogunErrc.h>

using namespace std;
using namespace shogun::io;

CBufferedInputStream::CBufferedInputStream(CInputStream* is, size_t buffer_bytes):
	CInputStream(),
	m_is(is),
	m_size(buffer_bytes)
{
	m_buffer.reserve(m_size);
}

CBufferedInputStream::~CBufferedInputStream()
{

}

error_condition CBufferedInputStream::read(string* buffer, int64_t size)
{
	if (size < 0)
		return make_error_condition(errc::invalid_argument);

	buffer->clear();
	if (m_status && size > 0)
		return m_status;

	buffer->reserve(size);

	error_condition r;
	while (buffer->size() < static_cast<size_t>(size))
	{
		// check the buffer
		if (m_pos == m_limit)
		{
			r = fill();
			// If we didn't read any bytes, we're at the end of the file; break out.
			if (m_limit == 0)
			{
				m_status = r;
				break;
			}
		}

		const int64_t bytes_to_copy =
			std::min<int64_t>(m_limit - m_pos, size - buffer->size());
		buffer->insert(buffer->size(), m_buffer, m_pos, bytes_to_copy);
		m_pos += bytes_to_copy;
	}
	if (io::is_out_of_range(r) &&
		(buffer->size() == static_cast<size_t>(size)))
		return {};

	return r;
}

error_condition CBufferedInputStream::skip(int64_t bytes)
{
	if (bytes < 0)
		return make_error_condition(errc::invalid_argument);

	if (m_pos + bytes < m_limit)
	{
		// If we aren't skipping too much, then we can just move pos_;
		m_pos += bytes;
	}
	else
	{
		auto r = m_is->skip(bytes - (m_limit - m_pos));
		m_pos = 0;
		m_limit = 0;
		if (io::is_out_of_range(r))
			m_status = r;
		return r;
	}
	return {};
}

int64_t CBufferedInputStream::tell() const
{
	return m_is->tell() - (m_limit - m_pos);
}

void CBufferedInputStream::reset()
{
	m_is->reset();
	m_pos = 0;
	m_limit = 0;
	m_status = {};
}


error_condition CBufferedInputStream::read_line(std::string* result)
{
	result->clear();
	error_condition r;
	while (true)
	{
		if (m_pos == m_limit)
		{
			// Get more data into buffer
			r = fill();
			if (m_limit == 0)
				break;
		}
		char c = m_buffer[m_pos++];
		if (c == '\n')
		{
		//if (include_eol)
		//	*result += c;

	  		return {};
		}
		// We don't append '\r' to *result
		if (c != '\r')
			*result += c;

	}
	if (io::is_out_of_range(r) && !result->empty())
		return {};

	return r;
}

error_condition CBufferedInputStream::read_all(std::string* result)
{
	result->clear();
	error_condition r;
	while (!r)
	{
		r = fill();
		if (m_limit == 0)
			break;

		result->append(m_buffer);
		m_pos = m_limit;
	}

	if (io::is_out_of_range(m_status))
	{
		m_status = r;
		return {};
	}
	return r;
}

error_condition CBufferedInputStream::fill()
{
	if (m_status)
	{
		m_pos = 0;
		m_limit = 0;
		return m_status;
	}

	auto r = m_is->read(&m_buffer, m_size);
	m_pos = 0;
	m_limit = m_buffer.size();
	if (m_buffer.empty())
	{
		m_status = r;
	}
	return r;
}
