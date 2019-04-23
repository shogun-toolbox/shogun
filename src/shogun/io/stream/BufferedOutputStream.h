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
	IGNORE_IN_CLASSLIST class BufferedOutputStream : public OutputStream
	{
	public:
		/**
		 * Construct a buffered output stream
		 *
		 * @param os
		 * @param buffer_bytes
		 */
		BufferedOutputStream(std::shared_ptr<OutputStream> os, index_t buffer_bytes = 4096):
			OutputStream(), m_os(std::move(os))
		{

		}

		BufferedOutputStream(BufferedOutputStream&& src):
			OutputStream(), m_os(std::move(src.m_os))
		{
			src.m_os = nullptr;
		}

		BufferedOutputStream& operator=(BufferedOutputStream&& src)
		{
			m_os = std::move(src.m_os);
			return *this;
		}

		~BufferedOutputStream() override
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
		std::shared_ptr<OutputStream> m_os;

		SG_DELETE_COPY_AND_ASSIGN(BufferedOutputStream);
	};
}

#endif /* __BUFFERED_OUTPUT_STREAM_H__ */
