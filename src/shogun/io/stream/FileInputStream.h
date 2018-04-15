/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vitktor Gal
 */
#ifndef __FILE_INPUT_STREAM_H__
#define __FILE_INPUT_STREAM_H__

#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/stream/InputStream.h>

#include <shogun/io/ShogunErrc.h>

namespace shogun
{
	namespace io
	{
#define IGNORE_IN_CLASSLIST

		IGNORE_IN_CLASSLIST class CFileInputStream : public CInputStream
		{
		public:
			CFileInputStream(RandomAccessFile* src, bool free = false);
			~CFileInputStream() override;

			std::error_condition read(std::string* buffer, int64_t size) override;

			std::error_condition skip(int64_t bytes) override;
			int64_t tell() const override;
			void reset() override;

			const char* get_name() const override { return "FileInputStream"; }

		private:
			RandomAccessFile* m_src;
			bool m_free;
			int64_t m_pos;

			SG_DELETE_COPY_AND_ASSIGN(CFileInputStream);
		};
	}
}

#endif
