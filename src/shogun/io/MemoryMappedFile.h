/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Yuyu Zhang, Viktor Gal, 
 *          Evan Shelhamer, Bjoern Esser, Roman Votyakov
 */

#ifndef __MEMORYMAPPEDFILE_H__
#define __MEMORYMAPPEDFILE_H__

#include <shogun/lib/config.h>

#include <shogun/io/SGIO.h>
#include <shogun/base/SGObject.h>

#include <stdio.h>
#include <string.h>
#ifndef _MSC_VER
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

namespace shogun
{
/** @brief memory mapped file
*
* Implements a memory mapped file for super fast file access.
*/
template <class T> class MemoryMappedFile : public SGObject
{
	public:
		/** default constructor  */
		MemoryMappedFile() :SGObject()
		{
			unstable(SOURCE_LOCATION);

			fd = 0;
			length = 0;
			address = NULL;
			rw = 'r';
			last_written_byte = 0;

			set_generic<T>();
		}

		/** constructor
		 *
		 * open a memory mapped file for read or read/write mode
		 *
		 * @param fname name of file, zero terminated string
		 * @param flag determines read or read write mode (can be 'r' or 'w')
		 * @param fsize overestimate of expected file size (in bytes)
		 *   when opened in write  mode; Underestimating the file size will
		 *   result in an error to occur upon writing. In case the exact file
		 *   size is known later on, it can be reduced via set_truncate_size()
		 *   before closing the file.
		 *
		 */
		MemoryMappedFile(const char* fname, char flag='r', int64_t fsize=0)
		: SGObject()
		{
			require(flag=='w' || flag=='r', "Only 'r' and 'w' flags are allowed");

			last_written_byte=0;
			rw=flag;

#ifdef _MSC_VER
			DWORD open_flags = GENERIC_READ;
			DWORD share_mode = FILE_SHARE_READ;
			DWORD create_disp = OPEN_EXISTING;
			DWORD mmap_prot = PAGE_READONLY;
			DWORD mmap_flags = FILE_MAP_READ;
			if (rw=='w')
			{
				open_flags |= GENERIC_WRITE;
				share_mode |= FILE_SHARE_WRITE;
				create_disp = OPEN_ALWAYS;
				mmap_prot = PAGE_READWRITE;
				mmap_flags = FILE_MAP_ALL_ACCESS;
			}

			fd = CreateFile(fname, open_flags, share_mode, 0, create_disp, FILE_ATTRIBUTE_NORMAL, NULL);
			if (rw=='w' && fsize)
			{
				LARGE_INTEGER desired_len;
				desired_len.QuadPart = fsize;
				uint8_t byte=0;
				DWORD bytes_written;
				if ((SetFilePointerEx(fd, desired_len, NULL, FILE_BEGIN) == 0) || (WriteFile(fd, &byte, 1, &bytes_written, NULL) == 0))
					error("Error creating file of size {} bytes", fsize);
			}

			DWORD length = GetFileSize(fd, NULL);
			if (length == INVALID_FILE_SIZE)
				error("Error determining file size");

			mapping = CreateFileMapping(fd, 0, mmap_prot, 0, 0, 0);

			address = MapViewOfFile(mapping, mmap_flags, 0, 0, length);
			if (address == NULL)
				error("Error mapping file");
#else
			int open_flags=O_RDONLY;
			int mmap_prot=PROT_READ;
			int mmap_flags=MAP_PRIVATE;

			if (rw=='w')
			{
				open_flags=O_RDWR | O_CREAT;
				mmap_prot=PROT_READ|PROT_WRITE;
				mmap_flags=MAP_SHARED;
			}

			fd = open(fname, open_flags, S_IRWXU | S_IRWXG | S_IRWXO);
			if (fd == -1)
				error("Error opening file");

			if (rw=='w' && fsize)
			{
				uint8_t byte=0;
				if (lseek(fd, fsize, SEEK_SET) != fsize || write(fd, &byte, 1) != 1)
					error("Error creating file of size {} bytes", fsize);
			}

			struct stat sb;
			if (fstat(fd, &sb) == -1)
				error("Error determining file size");

			length = sb.st_size;
			address = mmap(NULL, length, mmap_prot, mmap_flags, fd, 0);
			if (address == MAP_FAILED)
				error("Error mapping file");
#endif
			set_generic<T>();
		}

		/** destructor */
		~MemoryMappedFile() override
		{
#ifdef _MSC_VER
			UnmapViewOfFile(address);
			CloseHandle(mapping);
			if (rw=='w' && last_written_byte)
			{
				LARGE_INTEGER desired_len;
				desired_len.QuadPart = last_written_byte;
				if ((SetFilePointerEx(fd, desired_len, NULL, FILE_BEGIN) == 0) || (SetEndOfFile(fd) == 0)) {
					CloseHandle(fd);
					error("Error Truncating file to {} bytes", last_written_byte);
				}
			}
			CloseHandle(fd);
#else
			munmap(address, length);
			if (rw=='w' && last_written_byte && ftruncate(fd, last_written_byte) == -1)

			{
				close(fd);
				error("Error Truncating file to {} bytes", last_written_byte);
			}
			close(fd);
#endif
		}

		/** get the mapping address
		 * It can now be accessed via, e.g.
		 *
		 * double* x = get_map()
		 * x[index]= foo; (for write mode)
		 * foo = x[index]; (for read and write mode)
		 *
		 * @return length of file
		 */
		inline T* get_map()
		{
			return (T*) address;
		}

		/** get the number of objects of type T cointained in the file
		 *
		 * @return length of file
		 */
		uint64_t get_length()
		{
			return length/sizeof(T);
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
		char* get_line(uint64_t& len, uint64_t& offs)
		{
			char* s = (char*) address;
			for (uint64_t i=offs; i<length; i++)
			{
				if (s[i] == '\n')
				{
					char* line=&s[offs];
					len=i-offs;
					offs=i+1;
					return line;
				}
			}

			len=0;
			offs=length;
			return NULL;
		}

		/** write line to file
		 *
		 * @param line string to be written (must not contain '\n' and not
		 *									required to be zero terminated)
		 * @param len length of the string to be written
		 * @param offs offset to be passed for writing next line, should be 0
		 *			initially (returned via reference)
		 *
		 * @return line (NOT ZERO TERMINATED)
		 */
		void write_line(const char* line, uint64_t len, uint64_t& offs)
		{
			char* s = ((char*) address) + offs;
			if (len+1+offs > length)
				error("Writing beyond size of file");

			for (uint64_t i=0; i<len; i++)
				s[i] = line[i];

			s[len]='\n';
			offs+=length+1;
			last_written_byte=offs-1;
		}

		/** set file size
		 *
		 * When the file is opened for read/write mode, it will be truncated
		 * upon destruction of the MemoryMappedFile object. This is
		 * automagically determined when writing lines, but might have to be
		 * set manually for other data types, which is what this function is for.
		 *
		 * @param sz byte number at which to truncate the file, zero to disable
		 * file truncation. Has an effect only when file is opened with in
		 * read/write mode 'w'
		 */
		inline void set_truncate_size(uint64_t sz=0)
		{
			last_written_byte=sz;
		}

		/** count the number of lines in a file
		 *
		 * @return number of lines
		 */
		int32_t get_num_lines()
		{
			char* s = (char*) address;
			int32_t linecount=0;
			for (uint64_t i=0; i<length; i++)
			{
				if (s[i] == '\n')
					linecount++;
			}

			return linecount;
		}

		/** operator overload for file read only access
		 *
		 * DOES NOT DO ANY BOUNDS CHECKING
		 *
		 * @param index index
		 * @return element at index
		 */
		inline T operator[](uint64_t index) const
		{
		  return ((T*)address)[index];
		}

		/** operator overload for file read only access
		 *
		 * DOES NOT DO ANY BOUNDS CHECKING
		 *
		 * @param index index
		 * @return element at index
		 */
		inline T operator[](int32_t index) const
		{
		  return ((T*)address)[index];
		}

		/** @return object name */
		const char* get_name() const override { return "MemoryMappedFile"; }

	protected:
		/** file descriptor */
#ifdef _MSC_VER
		HANDLE fd;
		HANDLE mapping;
#else
		int fd;
#endif
		/** size of file */
		uint64_t length;
		/** mapping address */
		void* address;
		/** mode */
		char rw;

		/** last_written_byte */
		uint64_t last_written_byte;
};
}
#endif
