/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef __PARSEBUFFER_H__
#define __PARSEBUFFER_H__

#include <shogun/lib/config.h>

#if defined(HAVE_CXX11) || defined(HAVE_PTHREAD)

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/DataType.h>
#ifdef HAVE_CXX11
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
#elif HAVE_PTHREAD
#include <pthread.h>
#endif

namespace shogun
{

/// Specifies whether location is empty,
/// contains an unused example or a used example.
enum E_IS_EXAMPLE_USED
{
	E_EMPTY = 1,
	E_NOT_USED = 2,
	E_USED = 3
};

/** @brief Class Example is the container type for
 * the vector+label combination.
 *
 * The vector is stored as an SGVector<T>, and the
 * label is a float64_t.
 *
 * Objects of this type are stored in the ring of
 * class CParseBuffer.
 */
template <class T>
class Example
{
public:
	/// Label
	float64_t label;
	/// Feature vector of type T
	T* fv;
	index_t length;
};

/** @brief Class CParseBuffer implements a ring of
 * examples of a defined size. The ring stores
 * objects of the Example type.
 *
 * The feature vector and label are provided to this
 * class from an external source. CParseBuffer is
 * only responsible for managing how these examples
 * are represented and stored in the memory. It is
 * also responsible for returning stored examples to be
 * used by an external caller, and freeing up the space
 * when the example is used to make room for another
 * example to take its place.
 *
 * Writing of examples is done into whichever position
 * in the ring is free to be overwritten, or empty.
 */
template <class T> class CParseBuffer: public CSGObject
{
public:
	/**
	 * Constructor, taking buffer size as argument.
	 *
	 * @param size Ring size as number of examples
	 */
	CParseBuffer(int32_t size = 1024);

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
#ifdef HAVE_CXX11
		std::unique_lock<std::mutex> write_lk(*write_mutex, std::defer_lock);
		std::unique_lock<std::mutex> current_ex_lock(*ex_in_use_mutex[ex_write_index], std::defer_lock);
		std::lock(write_lk, current_ex_lock);
		while (ex_used[ex_write_index] == E_NOT_USED)
			ex_in_use_cond[ex_write_index]->wait(current_ex_lock);
		Example<T>* ex=&ex_ring[ex_write_index];
#elif HAVE_PTHREAD
		pthread_mutex_lock(write_lock);
		pthread_mutex_lock(&ex_in_use_mutex[ex_write_index]);
		while (ex_used[ex_write_index] == E_NOT_USED)
			pthread_cond_wait(&ex_in_use_cond[ex_write_index], &ex_in_use_mutex[ex_write_index]);
		Example<T>* ex=&ex_ring[ex_write_index];
		pthread_mutex_unlock(&ex_in_use_mutex[ex_write_index]);
		pthread_mutex_unlock(write_lock);
#endif

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
	Example<T>* return_example_to_read();

	/**
	 * Returns the next example from the buffer if unused, or NULL.
	 *
	 * @return unused example object at next 'read' position or NULL.
	 */
	Example<T>* get_unused_example();

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
	 * @param free_after_release whether to SG_FREE() the vector or not
	 */
	void finalize_example(bool free_after_release);

	/**
	 * Set whether all vectors are to be freed
	 * on destruction. This is true by default.
	 *
	 * The features object should set this to false if
	 * the vectors are freed manually later.
	 *
	 * @param destroy free examples on destruction or not
	 */
	void set_free_vectors_on_destruct(bool destroy)
	{
		free_vectors_on_destruct = destroy;
	}

	/**
	 * Return whether all example objects will be freed
	 * on destruction.
	 */
	bool get_free_vectors_on_destruct()
	{
		return free_vectors_on_destruct;
	}

	/**
	 * Return the name of the object
	 *
	 * @return ParseBuffer
	 */
	virtual const char* get_name() const { return "ParseBuffer"; }

	/** Initialize vector if free_vectors_on_destruct is True and the vector is NULL
	 *
	 */
	void init_vector();

protected:
	/**
	 * Increments the 'read' position in the buffer.
	 *
	 */
	virtual void inc_read_index()
	{
		ex_read_index=(ex_read_index + 1) % ring_size;
	}

	/**
	 * Increments the 'write' position in the buffer.
	 *
	 */
	virtual void inc_write_index()
	{
		ex_write_index=(ex_write_index + 1) % ring_size;
	}

protected:

	/// Size of ring as number of examples
	int32_t ring_size;
	/// Ring of examples
	Example<T>* ex_ring;

	/// Enum used for representing used/unused/empty state of example
	E_IS_EXAMPLE_USED* ex_used;
#ifdef HAVE_CXX11
	/// Lock on state of example - used or unused
	std::vector<std::shared_ptr<std::mutex> > ex_in_use_mutex;
	/// Condition variable triggered when example is being/not being used
	std::vector<std::shared_ptr<std::condition_variable> > ex_in_use_cond;
	/// Lock for reading examples from the ring
	std::shared_ptr<std::mutex> read_mutex;
	/// Lock for writing new examples
	std::shared_ptr<std::mutex> write_mutex;
#elif HAVE_PTHREAD
	/// Lock on state of example - used or unused
	pthread_mutex_t* ex_in_use_mutex;
	/// Condition variable triggered when example is being/not being used
	pthread_cond_t* ex_in_use_cond;
	/// Lock for reading examples from the ring
	pthread_mutex_t* read_lock;
	/// Lock for writing new examples
	pthread_mutex_t* write_lock;
#endif

	/// Write position for next example
	int32_t ex_write_index;
	/// Position of next example to be read
	int32_t ex_read_index;

	/// Whether examples on the ring will be freed on destruction
	bool free_vectors_on_destruct;
};


template <class T> void CParseBuffer<T>::init_vector()
{
	if (!free_vectors_on_destruct)
		return;
	for (int32_t i=0; i<ring_size; i++)
	{
		if(ex_ring[i].fv==NULL)
			ex_ring[i].fv = new T();
	}
}

template <class T> CParseBuffer<T>::CParseBuffer(int32_t size)
{
	ring_size = size;
	ex_ring = SG_CALLOC(Example<T>, ring_size);
	ex_used = SG_MALLOC(E_IS_EXAMPLE_USED, ring_size);
#ifdef HAVE_CXX11
	read_mutex = std::make_shared<std::mutex>();
	write_mutex = std::make_shared<std::mutex>();
#elif HAVE_PTHREAD
	ex_in_use_mutex = SG_MALLOC(pthread_mutex_t, ring_size);
	ex_in_use_cond = SG_MALLOC(pthread_cond_t, ring_size);
	read_lock = SG_MALLOC(pthread_mutex_t, 1);
	write_lock = SG_MALLOC(pthread_mutex_t, 1);
#endif

	SG_SINFO("Initialized with ring size: %d.\n", ring_size)

	ex_write_index = 0;
	ex_read_index = 0;

	for (int32_t i=0; i<ring_size; i++)
	{
		ex_used[i] = E_EMPTY;

		ex_ring[i].fv = NULL;
		ex_ring[i].length = 1;
		ex_ring[i].label = FLT_MAX;

#ifdef HAVE_CXX11
		ex_in_use_mutex.push_back(std::make_shared<std::mutex>());
		ex_in_use_cond.push_back(std::make_shared<std::condition_variable>());
#elif defined(HAVE_PTHREAD)
		pthread_cond_init(&ex_in_use_cond[i], NULL);
		pthread_mutex_init(&ex_in_use_mutex[i], NULL);
#endif
	}
#if defined(HAVE_PTHREAD) && !defined(HAVE_CXX11)
	pthread_mutex_init(read_lock, NULL);
	pthread_mutex_init(write_lock, NULL);
#endif

	free_vectors_on_destruct = true;
}

template <class T> CParseBuffer<T>::~CParseBuffer()
{
	for (int32_t i=0; i<ring_size; i++)
	{
		if (ex_ring[i].fv != NULL && free_vectors_on_destruct)
		{
			SG_DEBUG("%s::~%s(): destroying examples ring vector %d at %p\n",
					get_name(), get_name(), i, ex_ring[i].fv);
			delete ex_ring[i].fv;
		}
#if defined(HAVE_PTHREAD) && !defined(HAVE_CXX11)
		pthread_mutex_destroy(&ex_in_use_mutex[i]);
		pthread_cond_destroy(&ex_in_use_cond[i]);
#endif
	}
	SG_FREE(ex_ring);
	SG_FREE(ex_used);
#ifdef HAVE_CXX11
	ex_in_use_mutex.clear();
	ex_in_use_cond.clear();
	read_mutex.reset();
	write_mutex.reset();
#elif HAVE_PTHREAD
	SG_FREE(ex_in_use_mutex);
	SG_FREE(ex_in_use_cond);

	SG_FREE(read_lock);
	SG_FREE(write_lock);
#endif
}

template <class T>
int32_t CParseBuffer<T>::write_example(Example<T> *ex)
{
	ex_ring[ex_write_index].label = ex->label;
	ex_ring[ex_write_index].fv = ex->fv;
	ex_ring[ex_write_index].length = ex->length;
	ex_used[ex_write_index] = E_NOT_USED;
	inc_write_index();

	return 1;
}

template <class T>
Example<T>* CParseBuffer<T>::return_example_to_read()
{
	if (ex_read_index >= 0)
		return &ex_ring[ex_read_index];
	else
		return NULL;
}

template <class T>
Example<T>* CParseBuffer<T>::get_unused_example()
{
#ifdef HAVE_CXX11
	std::lock_guard<std::mutex> read_lk(*read_mutex);
#elif HAVE_PTHREAD
	pthread_mutex_lock(read_lock);
#endif

	Example<T> *ex;
	int32_t current_index = ex_read_index;
	// Because read index will change after return_example_to_read

#ifdef HAVE_CXX11
	std::lock_guard<std::mutex> current_ex_lk(*ex_in_use_mutex[current_index]);
#elif HAVE_PTHREAD
	pthread_mutex_lock(&ex_in_use_mutex[current_index]);
#endif

	if (ex_used[current_index] == E_NOT_USED)
		ex = return_example_to_read();
	else
		ex = NULL;

#if defined(HAVE_PTHREAD) && !defined(HAVE_CXX11)
	pthread_mutex_unlock(&ex_in_use_mutex[current_index]);
	pthread_mutex_unlock(read_lock);
#endif
	return ex;
}

template <class T>
int32_t CParseBuffer<T>::copy_example(Example<T> *ex)
{
#ifdef HAVE_CXX11
	std::lock_guard<std::mutex> write_lk(*write_mutex);
#elif HAVE_PTHREAD
	pthread_mutex_lock(write_lock);
#endif
	int32_t ret;
	int32_t current_index = ex_write_index;

#ifdef HAVE_CXX11
	std::unique_lock<std::mutex> current_ex_lock(*ex_in_use_mutex[current_index]);
#elif HAVE_PTHREAD
	pthread_mutex_lock(&ex_in_use_mutex[current_index]);
#endif
	while (ex_used[ex_write_index] == E_NOT_USED)
	{
#ifdef HAVE_CXX11
		ex_in_use_cond[ex_write_index]->wait(current_ex_lock);
#elif HAVE_PTHREAD
		pthread_cond_wait(&ex_in_use_cond[ex_write_index], &ex_in_use_mutex[ex_write_index]);
#endif
	}

	ret = write_example(ex);

#if defined(HAVE_PTHREAD) && !defined(HAVE_CXX11)
	pthread_mutex_unlock(&ex_in_use_mutex[current_index]);
	pthread_mutex_unlock(write_lock);
#endif

	return ret;
}

template <class T>
void CParseBuffer<T>::finalize_example(bool free_after_release)
{
#ifdef HAVE_CXX11
	std::lock_guard<std::mutex> read_lk(*read_mutex);
	std::unique_lock<std::mutex> current_ex_lock(*ex_in_use_mutex[ex_read_index]);
#elif HAVE_PTHREAD
	pthread_mutex_lock(read_lock);
	pthread_mutex_lock(&ex_in_use_mutex[ex_read_index]);
#endif
	ex_used[ex_read_index] = E_USED;

	if (free_after_release)
	{
		SG_DEBUG("Freeing object in ring at index %d and address: %p.\n",
			 ex_read_index, ex_ring[ex_read_index].fv);

		SG_FREE(ex_ring[ex_read_index].fv);
		ex_ring[ex_read_index].fv=NULL;
	}

#ifdef HAVE_CXX11
	ex_in_use_cond[ex_read_index]->notify_one();
	current_ex_lock.unlock();
#elif HAVE_PTHREAD
	pthread_cond_signal(&ex_in_use_cond[ex_read_index]);
	pthread_mutex_unlock(&ex_in_use_mutex[ex_read_index]);
#endif
	inc_read_index();

#if defined(HAVE_PTHREAD) && !defined(HAVE_CXX11)
	pthread_mutex_unlock(read_lock);
#endif
}

}
#endif // defined(HAVE_CXX11) || defined(HAVE_PTHREAD)
#endif // __PARSEBUFFER_H__
