/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vitktor Gal
 */
#ifndef __BYTE_ARRAY_INPUT_STREAM_H__
#define __BYTE_ARRAY_INPUT_STREAM_H__

#include <shogun/io/ShogunErrc.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/stream/InputStream.h>

#include <string_view>

namespace shogun
{
	namespace io
	{
#define IGNORE_IN_CLASSLIST

		IGNORE_IN_CLASSLIST class ByteArrayInputStream : public InputStream
		{
		public:
			ByteArrayInputStream(const char* _buffer, const size_t _length):
				InputStream(),	m_buffer(_buffer, _length) {}

			ByteArrayInputStream(const std::string& _buffer):
				ByteArrayInputStream(_buffer.c_str(), _buffer.length()) {}

			~ByteArrayInputStream() override {}

			std::error_condition read(std::string* buffer, int64_t size) override
			{
				if (size < 0)
					return make_error_condition(std::errc::invalid_argument);

				std::error_condition ec;
				if ((m_pos+size) > m_buffer.size())
				{
					size = m_buffer.size() - m_pos;
					ec = make_error_condition(ShogunErrc::OutOfRange);
				}
				buffer->clear();
				buffer->resize(size);
				buffer->assign(m_buffer.begin()+m_pos, m_buffer.begin()+m_pos+size);
				m_pos += size;
				return ec;
			}

			std::error_condition skip(int64_t bytes) override
			{
				if (bytes < 0)
					return make_error_condition(std::errc::invalid_argument);

				if ((m_pos + bytes) > m_buffer.size())
					return make_error_condition(ShogunErrc::OutOfRange);
				m_pos += bytes;
				return {};
			}

			int64_t tell() const override
			{
				return m_pos;
			}

			void reset() override
			{
				m_pos = 0;
			}

			const char* get_name() const override { return "ByteArrayInputStream"; }

		private:
			const std::string_view m_buffer;
			size_t m_pos = 0;

			SG_DELETE_COPY_AND_ASSIGN(ByteArrayInputStream);
		};
	}
}

#endif
