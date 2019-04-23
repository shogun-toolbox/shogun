/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vitktor Gal
 */
#ifndef __BUFFERED_INPUT_STREAM_H__
#define __BUFFERED_INPUT_STREAM_H__

#include <shogun/base/macros.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/stream/InputStream.h>

namespace shogun
{
	namespace io
	{
#define IGNORE_IN_CLASSLIST
		IGNORE_IN_CLASSLIST class BufferedInputStream : public InputStream
		{
		public:
			/**
			 * Construct a buffered output stream
			 *
			 * @param os
			 * @param buffer_bytes
			 */
			BufferedInputStream(InputStream* is, size_t buffer_bytes = 4096);

			~BufferedInputStream() override;

			std::error_condition read(std::string* buffer, int64_t size) override;
			std::error_condition skip(int64_t bytes) override;
			int64_t tell() const override;
			void reset() override;

			std::error_condition read_line(std::string* result);

			std::error_condition read_all(std::string* result);

			const char* get_name() const override
			{
				return "BufferedInputStream";
			}

		private:
			std::error_condition fill();
		private:
			InputStream* m_is;
			size_t m_size;
			std::string m_buffer;
			size_t m_pos = 0;
			size_t m_limit = 0;
			std::error_condition m_status;

			SG_DELETE_COPY_AND_ASSIGN(BufferedInputStream);
		};
	}
}

#endif /* __BUFFERED_OUTPUT_STREAM_H__ */
