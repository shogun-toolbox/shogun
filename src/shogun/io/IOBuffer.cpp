/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Shashwat Lal Das, Thoralf Klein, 
 *          Heiko Strathmann, Viktor Gal
 */

#include <string.h>
#include <fcntl.h>
#include <stdio.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <shogun/io/IOBuffer.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/v_array.h>

using namespace shogun;

IOBuffer::IOBuffer()
{
	init();
}

IOBuffer::IOBuffer(int fd)
{
	init();
	working_file = fd;
}

IOBuffer::~IOBuffer()
{
}

void IOBuffer::init()
{
	size_t s = 1 << 16;
	space.reserve(s);
	endloaded = space.begin;
	working_file=-1;
}

void IOBuffer::use_file(int fd)
{
	working_file = fd;
}

int IOBuffer::open_file(const char* name, char flag)
{
	int ret=1;
	switch(flag)
	{
	case 'r':
		working_file = open(name, O_RDONLY|O_LARGEFILE);
		break;

	case 'w':
		working_file = open(name, O_CREAT|O_TRUNC|O_WRONLY, 0666);
		break;

	default:
		error("Unknown file operation. Something other than 'r'/'w' specified.");
		ret = 0;
	}
	return ret;
}

void IOBuffer::reset_file()
{
	lseek(working_file, 0, SEEK_SET);
	endloaded = space.begin;
	space.end = space.begin;
}

void IOBuffer::set(char *p)
{
	space.end = p;
}

ssize_t IOBuffer::read_file(void* buf, size_t nbytes)
{
	return read(working_file, buf, nbytes);
}

size_t IOBuffer::fill()
{
	if (space.end_array - endloaded == 0)
	{
		size_t offset = endloaded - space.begin;
		space.reserve(2 * (space.end_array - space.begin));
		endloaded = space.begin+offset;
	}
	ssize_t num_read = read_file(endloaded, space.end_array - endloaded);
	if (num_read >= 0)
	{
		endloaded = endloaded+num_read;
		return num_read;
	}
	else
		return 0;
}

ssize_t IOBuffer::write_file(const void* buf, size_t nbytes)
{
	return write(working_file, buf, nbytes);
}

void IOBuffer::flush()
{
	if (working_file>=0)
	{
		if (write_file(space.begin, space.index()) != (int) space.index())
			error("Error, failed to write example!");
	}
	space.end = space.begin;
#ifdef _WIN32
	_commit(working_file);
#else
	fsync(working_file);
#endif
}

bool IOBuffer::close_file()
{
	if (working_file < 0)
		return false;
	else
	{
		int r = close(working_file);
		if (r < 0)
			error("Error closing the file!");
		return true;
	}
}

ssize_t IOBuffer::readto(char* &pointer, char terminal)
{
//Return a pointer to the bytes before the terminal.  Must be less
//than the buffer size.
	pointer = space.end;
	while (pointer != endloaded && *pointer != terminal)
		pointer++;
	if (pointer != endloaded)
	{
		size_t n = pointer - space.end;
		space.end = pointer+1;
		pointer -= n;
		return n;
	}
	else
	{
		if (endloaded == space.end_array)
		{
			size_t left = endloaded - space.end;
			memmove(space.begin, space.end, left);
			space.end = space.begin;
			endloaded = space.begin+left;
			pointer = endloaded;
		}
		if (fill() > 0)// more bytes are read.
			return readto(pointer,terminal);
		else //no more bytes to read, return nothing.
			return 0;
	}
}

void IOBuffer::buf_write(char* &pointer, int n)
{
	if (space.end + n <= space.end_array)
	{
		pointer = space.end;
		space.end += n;
	}
	else // Time to dump the file
	{
		if (space.end != space.begin)
			flush();
		else // Array is short, so increase size.
		{
			space.reserve(2 * (space.end_array - space.begin));
			endloaded = space.begin;
		}
		buf_write(pointer,n);
	}
}

unsigned int IOBuffer::buf_read(char* &pointer, int n)
{
	// Return a pointer to the next n bytes.
	// n must be smaller than the maximum size.
	if (space.end + n <= endloaded)
	{
		pointer = space.end;
		space.end += n;
		return n;
	}
	else // out of bytes, so refill.
	{
		if (space.end != space.begin) //There exists room to shift.
		{
			// Out of buffer so swap to beginning.
			int left = endloaded - space.end;
			memmove(space.begin, space.end, left);
			space.end = space.begin;
			endloaded = space.begin+left;
		}
		if (fill() > 0)
			return buf_read(pointer,n);// more bytes are read.
		else
		{
			// No more bytes to read, return all that we have left.
			pointer = space.end;
			space.end = endloaded;
			return endloaded - pointer;
		}
	}
}
