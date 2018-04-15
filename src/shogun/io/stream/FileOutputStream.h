/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vitktor Gal
 */
#ifndef __FILE_OUTPUT_STREAM_H__
#define __FILE_OUTPUT_STREAM_H__

#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/stream/OutputStream.h>

namespace shogun
{
#define IGNORE_IN_CLASSLIST

	IGNORE_IN_CLASSLIST class CFileOutputStream : public COutputStream
	{
	public:
		CFileOutputStream(std::unique_ptr<WritableFile> dest):
			COutputStream(), m_dest(std::move(dest)) {}
		~CFileOutputStream() override
		{
		}

		void close() override
		{
			m_dest->close();
		}

		void flush() override
		{
			m_dest->flush();
		}

		void write(const void* buffer, size_t size) override
		{
			m_dest->append(buffer, size);
		}

		const char* get_name() const override { return "FileOutputStream"; }

	private:
		std::unique_ptr<WritableFile> m_dest;

		SG_DELETE_COPY_AND_ASSIGN(CFileOutputStream);
	};
}

#endif
