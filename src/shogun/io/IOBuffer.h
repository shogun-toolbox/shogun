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

#ifndef IOBUFFER_H__
#define IOBUFFER_H__

#include <shogun/lib/v_array.h>
#include <shogun/lib/common.h>
#include <shogun/io/io.h>
#include <shogun/lib/DataType.h>
#include <shogun/base/SGObject.h>

#include <stdio.h>
#include <fcntl.h>
#include <iostream>

#ifndef O_LARGEFILE //for OSX
#define O_LARGEFILE 0
#endif

using namespace std;

namespace shogun
{
/** @brief An I/O buffer class.
 *
 * A file is read into buffer space, which is accessed through
 * extents; 'space.begin' is the start of the buffer, 'space.end' is the
 * address of the last read character.
 *
 * The buffer grows in size if required, the default size being 64KB.
 *
 */

 class CIOBuffer : public CSGObject
 {

 public:

	/** 
	 * Initialize the buffer, reserve 64K memory by default.
	 * 
	 */
	void init();

	/** 
	 * Open a file, in read or write mode.
	 * 
	 * @param name File name.
	 * @param flag CIOBuffer::READ or CIOBuffer::WRITE
	 * 
	 * @return 1 on success, 0 on error.
	 */
	virtual int open_file(const char* name, int flag=READ);

	/** 
	 * Seek back to zero, reset the buffer markers.
	 * 
	 */
	virtual void reset_file();

	/** 
	 * Constructor.
	 * 
	 */
	CIOBuffer();

	/** 
	 * Destructor.
	 * 
	 */
	~CIOBuffer();

	/** 
	 * Set the buffer marker to a position.
	 * 
	 * @param p Character pointer to which the end of buffer space is assigned.
	 */
	void set(char *p);

	/** 
	 * Read some bytes from the file into memory.
	 * 
	 * @param buf void* buffer into which to read.
	 * @param nbytes number of bytes to read
	 * 
	 * @return Number of bytes read successfully.
	 */
	virtual ssize_t read_file(void* buf, size_t nbytes);

	/** 
	 * Fill the buffer by reading as many bytes from the file as required.
	 * 
	 * @return Number of bytes read.
	 */
	size_t fill();

	/** 
	 * Write to the file from memory.
	 * 
	 * @param buf void* buffer from which to write.
	 * @param nbytes Number of bytes to write.
	 * 
	 * @return Number of bytes successfully written.
	 */
	virtual ssize_t write_file(const void* buf, size_t nbytes);

	/** 
	 * Flush the stream; commit all operations to file.
	 * 
	 */
	virtual void flush();

	/** 
	 * Close the file.
	 * 
	 * 
	 * @return true on success, false otherwise.
	 */
	virtual bool close_file();

	/** 
	 * Reads upto a terminal character from the buffer.
	 * 
	 * @param pointer Start of the string in the buffer, set by reference.
	 * @param terminal Terminal character upto which to read.
	 * 
	 * @return Number of characters read.
	 */
	size_t readto(char* &pointer, char terminal);

	virtual const char* get_name() const
	{
		return "IOBuffer";
	}
	

public:
  
	v_array<char> space; /**< space.begin = beginning of loaded
			      * values.  space.end = end of read or
			      * written values */
	
	char* endloaded; 	/**< end of loaded values */

	FILE* working_file;
	static const int READ = 1;
	static const int WRITE = 2;


};
}
#endif	/* IOBUFFER_H__ */
