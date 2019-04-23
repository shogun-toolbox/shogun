/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Viktor Gal, Soeren Sonnenburg, Yuyu Zhang,
 *          Sergey Lisitsyn, Wu Lin
 */
#ifndef __PARSEBUFFER_H__
#define __PARSEBUFFER_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/DataType.h>
#include <condition_variable>
#include <memory>
#include <mutex>
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
 * class ParseBuffer.
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

/** @brief Class ParseBuffer implements a ring of
 * examples of a defined size. The ring stores
 * objects of the Example type.
 *
 * The feature vector and label are provided to this
 * class from an external source. ParseBuffer is
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
template <class T> class ParseBuffer: public SGObject
{
public:
	/**
	 * Constructor, taking buffer size as argument.
	 *
	 * @param size Ring size as number of examples
	 */
	ParseBuffer(int32_t size = 1024);

	/**
	 * Destructor, frees up buffer.
	 *
	 */
	~ParseBuffer();

	/**
	 * Return the next position to write the example
	 * into the ring.
	 *
	 * @return pointer to example
	 */
	Example<T>* get_free_example()
	{
		std::unique_lock<std::mutex> write_lk(*write_mutex, std::defer_lock);
		std::unique_lock<std::mutex> current_ex_lock(*ex_in_use_mutex[ex_write_index], std::defer_lock);
		std::lock(write_lk, current_ex_lock);
		while (ex_used[ex_write_index] == E_NOT_USED)
			ex_in_use_cond[ex_write_index]->wait(current_ex_lock);
		Example<T>* ex=&ex_ring[ex_write_index];
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
	/// Lock on state of example - used or unused
	std::vector<std::shared_ptr<std::mutex> > ex_in_use_mutex;
	/// Condition variable triggered when example is being/not being used
	std::vector<std::shared_ptr<std::condition_variable> > ex_in_use_cond;
	/// Lock for reading examples from the ring
	std::shared_ptr<std::mutex> read_mutex;
	/// Lock for writing new examples
	std::shared_ptr<std::mutex> write_mutex;

	/// Write position for next example
	int32_t ex_write_index;
	/// Position of next example to be read
	int32_t ex_read_index;

	/// Whether examples on the ring will be freed on destruction
	bool free_vectors_on_destruct;
};


template <class T> void ParseBuffer<T>::init_vector()
{
	if (!free_vectors_on_destruct)
		return;
	for (int32_t i=0; i<ring_size; i++)
	{
		if(ex_ring[i].fv==NULL)
			ex_ring[i].fv = new T();
	}
}

template <class T> ParseBuffer<T>::ParseBuffer(int32_t size)
{
	ring_size = size;
	ex_ring = SG_CALLOC(Example<T>, ring_size);
	ex_used = SG_MALLOC(E_IS_EXAMPLE_USED, ring_size);
	read_mutex = std::make_shared<std::mutex>();
	write_mutex = std::make_shared<std::mutex>();
	SG_SINFO("Initialized with ring size: %d.\n", ring_size)

	ex_write_index = 0;
	ex_read_index = 0;

	for (int32_t i=0; i<ring_size; i++)
	{
		ex_used[i] = E_EMPTY;

		ex_ring[i].fv = NULL;
		ex_ring[i].length = 1;
		ex_ring[i].label = FLT_MAX;

		ex_in_use_mutex.push_back(std::make_shared<std::mutex>());
		ex_in_use_cond.push_back(std::make_shared<std::condition_variable>());
	}
	free_vectors_on_destruct = true;
}

template <class T> ParseBuffer<T>::~ParseBuffer()
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
	SG_FREE(ex_used);

	ex_in_use_mutex.clear();
	ex_in_use_cond.clear();
	read_mutex.reset();
	write_mutex.reset();
}

template <class T>
int32_t ParseBuffer<T>::write_example(Example<T> *ex)
{
	ex_ring[ex_write_index].label = ex->label;
	ex_ring[ex_write_index].fv = ex->fv;
	ex_ring[ex_write_index].length = ex->length;
	ex_used[ex_write_index] = E_NOT_USED;
	inc_write_index();

	return 1;
}

template <class T>
Example<T>* ParseBuffer<T>::return_example_to_read()
{
	if (ex_read_index >= 0)
		return &ex_ring[ex_read_index];
	else
		return NULL;
}

template <class T>
Example<T>* ParseBuffer<T>::get_unused_example()
{
	std::lock_guard<std::mutex> read_lk(*read_mutex);

	Example<T> *ex;
	int32_t current_index = ex_read_index;
	// Because read index will change after return_example_to_read

	std::lock_guard<std::mutex> current_ex_lk(*ex_in_use_mutex[current_index]);

	if (ex_used[current_index] == E_NOT_USED)
		ex = return_example_to_read();
	else
		ex = NULL;

	return ex;
}

template <class T>
int32_t ParseBuffer<T>::copy_example(Example<T> *ex)
{
	std::lock_guard<std::mutex> write_lk(*write_mutex);
	int32_t ret;
	int32_t current_index = ex_write_index;

	std::unique_lock<std::mutex> current_ex_lock(*ex_in_use_mutex[current_index]);
	while (ex_used[ex_write_index] == E_NOT_USED)
	{
		ex_in_use_cond[ex_write_index]->wait(current_ex_lock);
	}

	ret = write_example(ex);

	return ret;
}

template <class T>
void ParseBuffer<T>::finalize_example(bool free_after_release)
{
	std::lock_guard<std::mutex> read_lk(*read_mutex);
	std::unique_lock<std::mutex> current_ex_lock(*ex_in_use_mutex[ex_read_index]);
	ex_used[ex_read_index] = E_USED;

	if (free_after_release)
	{
		SG_DEBUG("Freeing object in ring at index %d and address: %p.\n",
			 ex_read_index, ex_ring[ex_read_index].fv);

		SG_FREE(ex_ring[ex_read_index].fv);
		ex_ring[ex_read_index].fv=NULL;
	}

	ex_in_use_cond[ex_read_index]->notify_one();
	current_ex_lock.unlock();
	inc_read_index();
}

}
#endif // __PARSEBUFFER_H__
