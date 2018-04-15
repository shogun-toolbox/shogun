/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vitktor Gal
 */
#ifndef __BUFFERED_OUTPUT_STREAM_H__
#define __BUFFERED_OUTPUT_STREAM_H__

#include <shogun/base/macros.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/stream/OutputStream.h>

namespace shogun
{
	IGNORE_IN_CLASSLIST class CBufferedOutputStream : public COutputStream
	{
	public:
		/**
		 * Construct a buffered output stream
		 *
		 * @param os
		 * @param buffer_bytes
		 */
		CBufferedOutputStream(std::shared_ptr<COutputStream> os, index_t buffer_bytes = 4096):
			COutputStream(), m_os(std::move(os))
		{

		}

		CBufferedOutputStream(CBufferedOutputStream&& src):
			COutputStream(), m_os(std::move(src.m_os))
		{
			src.m_os = nullptr;
		}

		CBufferedOutputStream& operator=(CBufferedOutputStream&& src)
		{
			m_os = std::move(src.m_os);
			return *this;
		}

		~CBufferedOutputStream() override
		{
			m_os->flush();
			m_os->close();
		}

		std::error_condition write(void* buffer, int64_t size) override
		{
			m_os->write(buffer, size);
		}

		std::error_condition close() override
		{
			m_os->close();
		}

		std::error_condition flush() override
		{
			m_os->flush();
		}

		const char* get_name() const override
		{
			return "BufferedOutputStream";
		}

	private:
		std::shared_ptr<COutputStream> m_os;

		SG_DELETE_COPY_AND_ASSIGN(CBufferedOutputStream);
	};
}

#endif /* __BUFFERED_OUTPUT_STREAM_H__ */
