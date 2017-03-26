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
#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/DataType.h>
#include <memory>
#include <vector>

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
		int32_t current_write_index = ex_write_index.load(std::memory_order_relaxed);  // only written from parser thread
		E_IS_EXAMPLE_USED old_value;
		while ((old_value=ex_used[current_write_index]->exchange(E_NOT_USED,std::memory_order_acq_rel) )== E_NOT_USED)
		{}
		Example<T>* ex=&ex_ring[ex_write_index];
		ex_used[current_write_index]->store(old_value,std::memory_order_relaxed);

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
		int32_t current_read_index  = ex_read_index.load(std::memory_order_relaxed); // only written from streaming thread
		ex_read_index.store((current_read_index + 1) % ring_size,std::memory_order_release);
	}

	/**
	 * Increments the 'write' position in the buffer.
	 *
	 */
	virtual void inc_write_index()
	{
		int32_t current_write_index = ex_write_index.load(std::memory_order_relaxed);  // only written from parser thread
		ex_write_index.store((current_write_index + 1) % ring_size,std::memory_order_release);
	}

protected:

	/// Size of ring as number of examples
	int32_t ring_size;
	/// Ring of examples
	Example<T>* ex_ring;

	/// Enum used for representing used/unused/empty state of example
	std::vector<std::shared_ptr<std::atomic<E_IS_EXAMPLE_USED> > > ex_used;

	/// Write position for next example
	std::atomic<int32_t> ex_write_index;
	/// Position of next example to be read
	std::atomic<int32_t> ex_read_index;

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

	SG_SINFO("Initialized with ring size: %d.\n", ring_size)

	ex_write_index = 0;
	ex_read_index = 0;

	for (int32_t i=0; i<ring_size; i++)
	{
		std::shared_ptr<std::atomic<E_IS_EXAMPLE_USED> > temp=std::make_shared<std::atomic<E_IS_EXAMPLE_USED> >(E_EMPTY);
		ex_used.push_back(temp);
		ex_ring[i].fv = NULL;
		ex_ring[i].length = 1;
		ex_ring[i].label = FLT_MAX;
	}

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
	}
	SG_FREE(ex_ring);
	ex_used.clear();

}

template <class T>
int32_t CParseBuffer<T>::write_example(Example<T> *ex)
{
	ex_ring[ex_write_index].label = ex->label;
	ex_ring[ex_write_index].fv = ex->fv;
	ex_ring[ex_write_index].length = ex->length;
	ex_used[ex_write_index]->store(E_NOT_USED,std::memory_order_release) ;
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
	Example<T> *ex;
	int32_t current_read_index = ex_read_index.load(std::memory_order_relaxed); // only written from streaming thread
	E_IS_EXAMPLE_USED old_value;
	if ((old_value=ex_used[current_read_index]->exchange(E_NOT_USED,std::memory_order_acq_rel)) == E_NOT_USED) //indirectly compared with current_write_index for checking available example
		ex = return_example_to_read();
	else
		ex = NULL;

	ex_used[current_read_index]->store(old_value,std::memory_order_relaxed);
	return ex;
}

template <class T>
int32_t CParseBuffer<T>::copy_example(Example<T> *ex)
{
	int32_t ret;
	int32_t current_write_index = ex_write_index.load(std::memory_order_relaxed);  // only written from parser thread
	// Because write index will change after write_example
	while (ex_used[current_write_index]->exchange(E_NOT_USED,std::memory_order_acq_rel) == E_NOT_USED) //indirectly compared with current_read_index for checking full buffer
	{}

	ret = write_example(ex);
	return ret;
}

template <class T>
void CParseBuffer<T>::finalize_example(bool free_after_release)
{
	int32_t current_read_index  = ex_read_index.load(std::memory_order_relaxed); // only written from streaming thread

	if (free_after_release)
	{
		SG_DEBUG("Freeing object in ring at index %d and address: %p.\n",
			 current_read_index, ex_ring[current_read_index].fv);

		SG_FREE(ex_ring[current_read_index].fv);
		ex_ring[current_read_index].fv=NULL;
	}

	ex_used[current_read_index]->store(E_USED,std::memory_order_release);
	inc_read_index();

}

}
#endif // __PARSEBUFFER_H__
