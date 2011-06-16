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
#include "lib/IOBuffer.h"

using namespace shogun;

void CIOBuffer::init()
{
	size_t s = 1 << 16;
	reserve(space, s);
	endloaded = space.begin;
}

int CIOBuffer::open_file(const char* name, int flag)
{
	int ret=1;
	switch(flag){
	case READ:
		working_file = fopen(name, "r");
		break;

	case WRITE:
		working_file = fopen(name, "w");
		break;

	default:
		SG_ERROR("Unknown file operation. Something other than READ/WRITE specified.\n");
		ret = 0;
	}
	return ret;
}

void CIOBuffer::reset_file()
{
	rewind(working_file);
	endloaded = space.begin;
	space.end = space.begin;
}

CIOBuffer::CIOBuffer()
{
	init();
}

CIOBuffer::~CIOBuffer()
{
	free(space.begin);
}

void CIOBuffer::set(char *p)
{
	space.end = p;
}

ssize_t CIOBuffer::read_file(void* buf, size_t nbytes)
{
	return fread(buf, 1, nbytes, working_file);
}

size_t CIOBuffer::fill()
{
	if (space.end_array - endloaded == 0)
	{
		size_t offset = endloaded - space.begin;
		reserve(space, 2 * (space.end_array - space.begin));
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
	return fwrite(buf, 1, nbytes, working_file);
}

void CIOBuffer::flush()
{
	if (write_file(space.begin, space.index()) != (int) space.index())
		SG_ERROR("Error, failed to write example!\n");
	space.end = space.begin;
	fflush(working_file);
}

bool CIOBuffer::close_file()
{
	if (working_file == NULL)
		return false;
	else
		return bool(fclose(working_file));
}

size_t CIOBuffer::readto(char* &pointer, char terminal)
{
//Return a pointer to the bytes before the terminal.  Must be less than the buffer size.
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
