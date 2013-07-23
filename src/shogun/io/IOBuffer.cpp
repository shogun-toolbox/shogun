/*
  Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
  embodied in the content of this file are licensed under the BSD
  (revised) open source license.

  Copyright (c) 2011 Berlin Institute of Technology and Max-Planck-Society.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  Shogun adjustments (w) 2011 Shashwat Lal Das
*/

#include <string.h>
#include <shogun/io/IOBuffer.h>

using namespace shogun;

CIOBuffer::CIOBuffer()
{
	init();
}

CIOBuffer::CIOBuffer(int fd)
{
	init();
	working_file = fd;
}

CIOBuffer::~CIOBuffer()
{
}

void CIOBuffer::init()
{
	size_t s = 1 << 16;
	space.reserve(s);
	endloaded = space.begin;
	working_file=-1;
}

void CIOBuffer::use_file(int fd)
{
	working_file = fd;
}

int CIOBuffer::open_file(const char* name, char flag)
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
		SG_ERROR("Unknown file operation. Something other than 'r'/'w' specified.\n")
		ret = 0;
	}
	return ret;
}

void CIOBuffer::reset_file()
{
	lseek(working_file, 0, SEEK_SET);
	endloaded = space.begin;
	space.end = space.begin;
}

void CIOBuffer::set(char *p)
{
	space.end = p;
}

ssize_t CIOBuffer::read_file(void* buf, size_t nbytes)
{
	return read(working_file, buf, nbytes);
}

size_t CIOBuffer::fill()
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

ssize_t CIOBuffer::write_file(const void* buf, size_t nbytes)
{
	return write(working_file, buf, nbytes);
}

void CIOBuffer::flush()
{
	if (working_file>=0)
	{
		if (write_file(space.begin, space.index()) != (int) space.index())
			SG_ERROR("Error, failed to write example!\n")
	}
	space.end = space.begin;
	fsync(working_file);
}

bool CIOBuffer::close_file()
{
	if (working_file < 0)
		return false;
	else
	{
		int r = close(working_file);
		if (r < 0)
			SG_ERROR("Error closing the file!\n")
		return true;
	}
}

ssize_t CIOBuffer::readto(char* &pointer, char terminal)
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

void CIOBuffer::buf_write(char* &pointer, int n)
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

unsigned int CIOBuffer::buf_read(char* &pointer, int n)
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
