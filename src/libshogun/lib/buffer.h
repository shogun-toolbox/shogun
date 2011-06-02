/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "lib/common.h"
#include <pthread.h>

#ifndef __PARSEBUFFER_H__
#define __PARSEBUFFER_H__


enum E_IS_EXAMPLE_USED
{
	E_EMPTY = 1,
	E_NOT_USED = 2,
	E_USED = 3
};

class example
{
public:
	float64_t label;
	int32_t len;
	float64_t* feature_vector;
};

class ParseBuffer
{
public:
	/** 
	 * Constructor, taking buffer size as argument.
	 * 
	 * @param size Buffer size in MB
	 */
	ParseBuffer(int32_t size);
	
	/** 
	 * Destructor, frees up buffer.
	 * 
	 */
	~ParseBuffer();

	/** 
	 * Writes the given example into the appropriate buffer space.
	 * Feature vector is copied into a separate block.
	 *
	 * @param ex Example to copy into buffer
	 * 
	 * @return 1 if successful, 0 on failure (if no space available)
	 */
	int32_t write_example(example* ex);
	
	/** 
	 * Returns the example that should be read next from the buffer.
	 * 
	 * @return example object at next 'read' position
	 */
	example* get_example();

	/** 
	 * Returns the next example from the buffer if unused, or NULL.
	 * 
	 * 
	 * @return unused example object at next 'read' position or NULL.
	 */
	example* fetch_example();

	/** 
	 * Copies an example into the buffer, waiting for the
	 * destination example to be used if necessary.
	 * 
	 * @param ex Example to copy into buffer
	 * 
	 * @return 1 on success, 0 on memory errors
	 */
	int32_t copy_example(example* ex);

	/** 
	 * Mark the example in 'read' position as 'used'.
	 * 
	 * It will then be free to be overwritten.
	 */
	void finalize_example();

protected:
	/** 
	 * Increments the 'read' position in the buffer.
	 * 
	 */
	virtual void inc_read_index();

	/** 
	 * Increments the 'write' position in the buffer.
	 * 
	 * @param len Length of feature vector
	 */
	virtual void inc_write_index(int32_t len);
	
protected:
	
	int32_t buffer_size;		/**< buffer size in bytes */
	example* ex_buff;			/**< buffer of example objects */
	float64_t* fv_buff;			/**< buffer space for feature vectors */

	E_IS_EXAMPLE_USED* ex_used;
	pthread_mutex_t* ex_in_use_mutex;
	pthread_cond_t* ex_in_use_cond;
	
	int32_t ex_write_index;		/**< write position for next example */
	int32_t ex_read_index;		/**< position of next example to be read */
	
	int32_t fv_write_index;		/**< write position for next feature vector */
	
};

#endif // __PARSEBUFFER_H__
