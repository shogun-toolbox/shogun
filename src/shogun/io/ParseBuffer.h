/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <pthread.h>

#ifndef __PARSEBUFFER_H__
#define __PARSEBUFFER_H__

namespace shogun
{

	enum E_IS_EXAMPLE_USED
	{
		E_EMPTY = 1,
		E_NOT_USED = 2,
		E_USED = 3
	};

	template <class T>
		class Example
	{
	public:
		float64_t label;
		SGVector<T> fv;
	};

	template <class T>
		class CParseBuffer
	{
	public:
		/** 
		 * Constructor, taking buffer size as argument.
		 * 
		 * @param size Ring size as number of examples
		 */
		CParseBuffer(int32_t size);
	
		/** 
		 * Destructor, frees up buffer.
		 * 
		 */
		~CParseBuffer();

		/**
		 * Return the next position to write the example
		 * into the ring.
		 *
		 * @return pointer to example
		 */
		Example<T>* get_free_example()
		{
			pthread_mutex_lock(write_lock);
			pthread_mutex_lock(&ex_in_use_mutex[ex_write_index]);
			while (ex_used[ex_write_index] == E_NOT_USED)
				pthread_cond_wait(&ex_in_use_cond[ex_write_index], &ex_in_use_mutex[ex_write_index]);
			Example<T>* ex=&ex_buff[ex_write_index];
			pthread_mutex_unlock(&ex_in_use_mutex[ex_write_index]);
			pthread_mutex_unlock(write_lock);

			return ex;
		}

		/** 
		 * Writes the given example into the appropriate buffer space.
		 * Feature vector is copied into a separate block.
		 *
		 * @param ex Example to copy into buffer
		 * 
		 * @return 1 if successful, 0 on failure (if no space available)
		 */
		int32_t write_example(Example<T>* ex);
	
		/** 
		 * Returns the example that should be read next from the buffer.
		 * 
		 * @return example object at next 'read' position
		 */
		Example<T>* get_example();

		/** 
		 * Returns the next example from the buffer if unused, or NULL.
		 * 
		 * @return unused example object at next 'read' position or NULL.
		 */
		Example<T>* fetch_example();

		/** 
		 * Copies an example into the buffer, waiting for the
		 * destination example to be used if necessary.
		 * 
		 * @param ex Example to copy into buffer
		 * 
		 * @return 1 on success, 0 on memory errors
		 */
		int32_t copy_example(Example<T>* ex);

		/** 
		 * Mark the example in 'read' position as 'used'.
		 * 
		 * It will then be free to be overwritten.
		 * 
		 * @param do_delete whether to SG_FREE() the vector or not
		 */
		void finalize_example(bool do_delete);

	protected:
		/** 
		 * Increments the 'read' position in the buffer.
		 * 
		 */
		inline virtual void inc_read_index()
		{
			ex_read_index=(ex_read_index + 1) % buffer_size;
		}

		/** 
		 * Increments the 'write' position in the buffer.
		 * 
		 */
		inline virtual void inc_write_index()
		{
			ex_write_index=(ex_write_index + 1) % buffer_size;
		}
	
	protected:
	
		int32_t buffer_size;		/**< buffer size as number of examples */
		Example<T>* ex_buff;			/**< buffer of example objects */

		E_IS_EXAMPLE_USED* ex_used;
		pthread_mutex_t* ex_in_use_mutex;
		pthread_cond_t* ex_in_use_cond;
		pthread_mutex_t* read_lock;
		pthread_mutex_t* write_lock;
	
		int32_t ex_write_index;		/**< write position for next example */
		int32_t ex_read_index;		/**< position of next example to be read */

		Example<T>* ex_buff_ring;
	};

	template <class T>
		CParseBuffer<T>::CParseBuffer(int32_t size)
	{
		buffer_size=size;

		ex_buff = SG_CALLOC(Example<T>, buffer_size);

		SG_SINFO("Initialized with ring size: %d.\n", buffer_size);
		ex_used = SG_MALLOC(E_IS_EXAMPLE_USED, buffer_size);
		ex_in_use_mutex = SG_MALLOC(pthread_mutex_t, buffer_size);
		ex_in_use_cond = SG_MALLOC(pthread_cond_t, buffer_size);
		read_lock = new pthread_mutex_t;
		write_lock = new pthread_mutex_t;
	
		ex_write_index = 0;
		ex_read_index = 0;

		for (int32_t i=0; i<buffer_size; i++)
		{
			ex_used[i] = E_EMPTY;
			ex_buff[i].fv.vector = new T();
			ex_buff[i].fv.vlen = 1;
			ex_buff[i].label = FLT_MAX;

			pthread_cond_init(&ex_in_use_cond[i], NULL);
			pthread_mutex_init(&ex_in_use_mutex[i], NULL);
		}
		pthread_mutex_init(read_lock, NULL);
		pthread_mutex_init(write_lock, NULL);
	}

	template <class T>
		CParseBuffer<T>::~CParseBuffer()
	{
		for (int32_t i=0; i<buffer_size; i++)
		{
			if (ex_buff[i].fv.vector != NULL)
				delete ex_buff[i].fv.vector;
			pthread_mutex_destroy(&ex_in_use_mutex[i]);
			pthread_cond_destroy(&ex_in_use_cond[i]);
		}
		SG_FREE(ex_buff);
		SG_FREE(ex_used);
		SG_FREE(ex_in_use_mutex);
		SG_FREE(ex_in_use_cond);

		delete read_lock;
		delete write_lock;
	}

	template <class T>
		int32_t CParseBuffer<T>::write_example(Example<T> *ex)
	{
		ex_buff[ex_write_index].label = ex->label;
		ex_buff[ex_write_index].fv.vector = ex->fv.vector;
		ex_buff[ex_write_index].fv.vlen = ex->fv.vlen;
		ex_used[ex_write_index] = E_NOT_USED;
		inc_write_index();

		return 1;	
	}

	template <class T>
		Example<T>* CParseBuffer<T>::get_example()
	{
		if (ex_read_index >= 0)
			return &ex_buff[ex_read_index];
		else
			return NULL;
	}

	template <class T>
		Example<T>* CParseBuffer<T>::fetch_example()
	{
		pthread_mutex_lock(read_lock);

		Example<T> *ex;
		int32_t current_index = ex_read_index;
		// Because read index will change after get_example

		pthread_mutex_lock(&ex_in_use_mutex[current_index]);
		
		if (ex_used[current_index] == E_NOT_USED)
			ex = get_example();
		else
			ex = NULL;

		pthread_mutex_unlock(&ex_in_use_mutex[current_index]);

		pthread_mutex_unlock(read_lock);
		return ex;
	}

	template <class T>
		int32_t CParseBuffer<T>::copy_example(Example<T> *ex)
	{
		pthread_mutex_lock(write_lock);
		int32_t ret;
		int32_t current_index = ex_write_index;

		pthread_mutex_lock(&ex_in_use_mutex[current_index]);
		while (ex_used[ex_write_index] == E_NOT_USED)
		{
			pthread_cond_wait(&ex_in_use_cond[ex_write_index], &ex_in_use_mutex[ex_write_index]);
		}
	
		ret=write_example(ex);

		pthread_mutex_unlock(&ex_in_use_mutex[current_index]);
		pthread_mutex_unlock(write_lock);

		return ret;
	}

	template <class T>
		void CParseBuffer<T>::finalize_example(bool do_delete)
	{
		pthread_mutex_lock(read_lock);
		pthread_mutex_lock(&ex_in_use_mutex[ex_read_index]);
		ex_used[ex_read_index] = E_USED;

		if (do_delete)
		{
			delete ex_buff[ex_read_index].fv.vector;
			ex_buff[ex_read_index].fv.vector=NULL;
		}

		pthread_cond_signal(&ex_in_use_cond[ex_read_index]);
		pthread_mutex_unlock(&ex_in_use_mutex[ex_read_index]);
		inc_read_index();

		pthread_mutex_unlock(read_lock);
	}

}
#endif // __PARSEBUFFER_H__
