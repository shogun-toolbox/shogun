/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Thoralf Klein, Heiko Strathmann, Yuyu Zhang, 
 *          Evan Shelhamer, Bjoern Esser
 */

#ifndef __BINARYSTREAM_H__
#define __BINARYSTREAM_H__

#include <shogun/lib/config.h>

#include <shogun/io/SGIO.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/memory.h>

#include <stdio.h>
#include <sys/stat.h>

namespace shogun
{
/** @brief memory mapped emulation via binary streams (files)
*
* Implements memory mapped file emulation (\sa MemoryMappedFile) via standard
* file operations like fseek, fread etc
*/
template <class T> class BinaryStream : public SGObject
{
public:
	/** default constructor
	 */
	BinaryStream() : SGObject()
	{
		rw = NULL;
		m_fname = NULL;
		fd = NULL;
		length = 0;

		set_generic<T>();
	}

	/** constructor
	 *
	 * open a file for read mode
	 *
	 * @param fname name of file, zero terminated string
	 * @param flag determines read or read write mode (currently only 'r' is supported)
	 */
	BinaryStream(const char * fname, const char * flag = "r")
		: SGObject()
	{
		/* open_stream(bs.m_fname, bs.rw); */
		SG_NOTIMPLEMENTED
		set_generic<T>();
	}


	/** copy constructor
	 *
	 * @param bs binary stream to copy from
	 */
	BinaryStream(const BinaryStream &bs)
	{
		open_stream(bs.m_fname, bs.rw);
		ASSERT(length == bs.length)
		set_generic<T>();
	}


	/** destructor */
	virtual ~BinaryStream()
	{
		close_stream();
	}

	/** open file stream
	 *
	 * @param fname file name
	 * @param flag flags "r" for reading etc
	 */
	void open_stream(const char * fname, const char * flag = "r")
	{
		rw = get_strdup(flag);
		m_fname = get_strdup(fname);

		fd = fopen(fname, flag);
		if (!fd)
			SG_ERROR("Error opening file '%s'\n", m_fname)

		struct stat sb;
		if (stat(fname, &sb) == -1)
			SG_ERROR("Error determining file size of '%s'\n", m_fname)

		length = sb.st_size;
		SG_DEBUG("Opened file '%s' of size %ld byte\n", fname, length)
	}

	/** close a file stream */
	void close_stream()
	{
		SG_FREE(rw);
		SG_FREE(m_fname);
		if (fd)
		{
			fclose(fd);
		}

		rw = NULL;
		m_fname = NULL;
		fd = NULL;
		length = 0;
	}

	/** get the number of objects of type T cointained in the file
	 *
	 * @return length of file
	 */
	uint64_t get_length()
	{
		return length / sizeof(T);
	}

	/** get the size of the file in bytes
	 *
	 * @return size of file in bytes
	 */
	uint64_t get_size()
	{
		return length;
	}

	/** get next line from file
	 *
	 * The returned line may be modfied in case the file was opened
	 * read/write. It is otherwise read-only.
	 *
	 * @param len length of line (returned via reference)
	 * @param offs offset to be passed for reading next line, should be 0
	 *			initially (returned via reference)
	 *
	 * @return line (NOT ZERO TERMINATED)
	 */
	char * get_line(uint64_t &len, uint64_t &offs)
	{
		return NULL;
	}

	/** count the number of lines in a file
	 *
	 * @return number of lines
	 */
	int32_t get_num_lines()
	{
		return 0;
	}

	/** read num elements starting from index into buffer
	 *
	 * @param buffer buffer that has to be at least num elements long
	 * @param index index into file starting from which elements are read
	 * @param num number of elements to be read
	 */
	void pre_buffer(T * buffer, long index, long num) const
	{
		ASSERT(index >= 0)
		ASSERT(num >= 0)

		if (num == 0)
		{
			return;
		}

		if (fseek(fd, ((long) sizeof(T)) * ((long) index), SEEK_SET) != 0)
			SG_ERROR("Error seeking to %ld (file '%s')\n", sizeof(T) * ((int64_t) index), m_fname)

			if (fread(buffer, sizeof(T), num, fd) != num)
				SG_ERROR("Error calling fread (file '%s')\n", m_fname)
			}

	/** read next
	 *
	 * @return next element
	 */
	inline T read_next() const
	{
		T ptr;
		if (fread(&ptr, sizeof(T), 1, fd) != 1)
		{
			fprintf(stderr, "Error calling fread (file '%s')\n", m_fname);
			exit(1);
		}
		return ptr;
	}

	/** operator overload for file read only access
	 *
	 * @param index index
	 * @return element at index
	 */
	inline T operator[](int32_t index) const
	{

		if (fseek(fd, ((long) sizeof(T)) * ((long) index), SEEK_SET) != 0)
			SG_ERROR("Error seeking to %ld (file '%s')\n", sizeof(T) * ((int64_t) index), m_fname)

		T ptr;

		if (fread(&ptr, sizeof(T), 1, fd) != 1)
			SG_ERROR("Error calling fread (file '%s')\n", m_fname)

		return ptr;
	}

	/** @return object name */
	virtual const char * get_name() const
	{
		return "BinaryStream";
	}

protected:
	/** file descriptor */
	FILE * fd;
	/** size of file */
	uint64_t length;
	/** mode */
	char * rw;
	/** fname */
	char * m_fname;
};
}
#endif // BINARY_STREAM
