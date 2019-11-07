/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vitktor Gal
 */
#ifndef __FILE_OUTPUT_STREAM_H__
#define __FILE_OUTPUT_STREAM_H__

#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/stream/OutputStream.h>

#include <string_view>

namespace shogun
{
	namespace io
	{
#define IGNORE_IN_CLASSLIST

		IGNORE_IN_CLASSLIST class FileOutputStream : public OutputStream
		{
		public:
			FileOutputStream(WritableFile* dest, bool free = false):
				OutputStream(), m_dst(dest), m_free(free) {}
			~FileOutputStream() override
			{
				if (m_free)
					delete m_dst;
			}

			std::error_condition close() override
			{
				return m_dst->close();
			}

			std::error_condition flush() override
			{
				return m_dst->flush();
			}

			std::error_condition write(const void* buffer, int64_t size) override
			{
				return m_dst->append(std::string_view(static_cast<const char*>(buffer), size));
			}

			const char* get_name() const override { return "FileOutputStream"; }

		private:
			WritableFile* m_dst;
			bool m_free;

			SG_DELETE_COPY_AND_ASSIGN(FileOutputStream);
		};
	}
}

#endif
