/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vitktor Gal
 */
#ifndef __BYTE_ARRAY_OUTPUT_STREAM_H__
#define __BYTE_ARRAY_OUTPUT_STREAM_H__

#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/stream/OutputStream.h>

#include <vector>

namespace shogun
{
	namespace io
	{
#define IGNORE_IN_CLASSLIST

		IGNORE_IN_CLASSLIST class ByteArrayOutputStream : public OutputStream
		{
		public:
			ByteArrayOutputStream(): OutputStream() {}
			~ByteArrayOutputStream() override {}

			std::error_condition close() override
			{
				 return {};
			}

			std::error_condition flush() override
			{
				 return {};
			}

			std::error_condition write(const void* buffer, int64_t size) override
			{
				std::copy(
					static_cast<const char*>(buffer),
					static_cast<const char*>(buffer) + size,
					std::back_inserter(m_byte_array));
				return {};
			}

			std::vector<char> content() const
			{
				return m_byte_array;
			}

			std::string as_string() const
			{
				return std::string(m_byte_array.begin(), m_byte_array.end());
			}

			const char* get_name() const override { return "ByteArrayOutputStream"; }

		private:
			std::vector<char> m_byte_array;

			SG_DELETE_COPY_AND_ASSIGN(ByteArrayOutputStream);
		};
	}
}

#endif
