/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013,2014 Thoralf Klein
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#ifndef __SGSPARSEVECTOR_H__
#define __SGSPARSEVECTOR_H__

#include <shogun/lib/config.h>

#include <shogun/lib/SGReferencedData.h>
#include <shogun/lib/SGVector.h>

#include <stdexcept>

namespace shogun
{
	class CFile;

/** @brief template class SGSparseVectorEntry */
template <class T> struct SGSparseVectorEntry
{
	/** feature index  */
	index_t feat_index;
	/** entry ... */
	T entry;
};

/** @brief template class SGSparseVector
 * The assumtion is that the stored SGSparseVectorEntry<T>* vector is ordered
 * by SGSparseVectorEntry.feat_index in non-decreasing order.
 * This has to be assured by the user of the class.
 */
template <class T> class SGSparseVector : public SGReferencedData
{
public:
	/** default constructor */
	SGSparseVector();

	/** constructor for setting params
	 *
	 * @param feats vector of SGSparseVectorEntry ordered by SGSparseVectorEntry.feat_index in non-decreasing order
	 * @param num_entries number of elements in feats vector
	 * @param ref_counting use reference counting
	 */
	SGSparseVector(SGSparseVectorEntry<T>* feats, index_t num_entries,
			bool ref_counting=true);

	/** constructor to create new vector in memory */
	SGSparseVector(index_t num_entries, bool ref_counting=true);

	/** copy constructor */
	SGSparseVector(const SGSparseVector& orig);

	virtual ~SGSparseVector();

	/** compute the dot product between dense weights and a sparse feature vector
	 * alpha * sparse^T * w + b
	 *
	 * possible with subset
	 *
	 * @param alpha scalar to multiply with
	 * @param vec dense vector to compute dot product with
	 * @param dim length of the dense vector
	 * @param b bias
	 * @return dot product between dense weights and a sparse feature vector
	 */
	T dense_dot(T alpha, T* vec, int32_t dim, T b);

	/** compute the dot product between dense weights and a sparse feature vector
	 * sparse^T * w
	 *
	 * @param vec dense vector to compute dot product with
	 * @return dot product between dense weights and a sparse feature vector
	 */
	template<typename ST> T dense_dot(SGVector<ST> vec);

	/** compute the dot product between current sparse vector and a given
	 * sparse vector.
	 * sparse_a^T * sparse_b
	 *
	 * @param v sparse vector
	 * @return dot product between the current sparse vector and v sparse vector
	 */
	T sparse_dot(const SGSparseVector<T>& v);

	/** compute the dot product between two sparse vectors.
	 * sparse_a^T * sparse_b
	 *
	 * @param a sparse vector
	 * @param b sparse vector
	 * @return dot product between a and b
	 */
	static T sparse_dot(const SGSparseVector<T>& a, const SGSparseVector<T>& b);

	/**
	 * get the sparse vector (no copying is done here)
	 *
	 * @return the refcount increased vector
	 */
	inline SGSparseVector<T> get()
	{
		return *this;
	}

	/**
	 * get number of dimensions
	 *
	 * @return largest feature index
	 */
	int32_t get_num_dimensions();

	/**
	 * sort features by indices  (Setting stable_pointer=true to
	 * guarantee that pointer features does not change. On the
	 * other hand, stable_pointer=false can shrink the vector if
	 * possible.)
	 *
	 * @param stable_pointer (default false) enforce stable pointer
	 */
	void sort_features(bool stable_pointer = false);

	/**
	 * Utility function to tell if feature indices are sorted
	 *
	 * @return bool (true if sorted, else false)
	 */
	bool is_sorted() const;

	/**
	 * get feature value for index
	 *
	 * @param index
	 * @return value
	 */
	T get_feature(int32_t index);

	/**
	 * get dense representation of given size
	 *
	 * @param dimension of requested dense vector
	 * @return SGVector<T>
	 */
	SGVector<T> get_dense(int32_t dimension);

	/**
	 * get shortet dense representation for sparse vector
	 *
	 * @return SGVector<T>
	 */
	SGVector<T> get_dense();

	/** clone vector */
	SGSparseVector<T> clone() const;

	/** load vector from file
	 *
	 * @param loader File object via which to load data
	 */
	void load(CFile* loader);

	/** save vector to file
	 *
	 * @param saver File object via which to save data
	 */
	void save(CFile* saver);

	/** add a sparse feature vector onto a dense one
	 * dense += alpha*sparse
	 *
	 * @param alpha scalar to multiply with
	 * @param vec dense vector
	 * @param dim length of the dense vector
	 * @param abs_val if true, do dense+=alpha*abs(sparse)
	 */
	void add_to_dense(T alpha, T * vec, int32_t dim, bool abs_val = false);

	/** display vector
	 *
	 * @param name   vector name in output
	 * @param prefix prepend on every entry
	 */
	void display_vector(const char* name="vector",
			const char* prefix="");

protected:
	virtual void copy_data(const SGReferencedData& orig);

	virtual void init_data();

	virtual void free_data();

	/** helper function to compute dot product for unsorted sparse vectors
	 *
	 * You should not use this at all, since computational complexity is in O(n^2)
	 *
	 * @param a vector a
	 * @param b vector b
	 *
	 * @return dot product
	 */
	static T dot_prod_expensive_unsorted(const SGSparseVector<T>& a, const SGSparseVector<T>& b);

public:
	/** number of feature entries */
	index_t num_feat_entries;

	/** features */
	SGSparseVectorEntry<T>* features;

};

template <typename S>
bool equals(SGSparseVector<S>* lhs, SGSparseVector<S>* rhs)
{
    throw std::logic_error("Equals not supported by SGSparseVector");
}

}

#endif // __SGSPARSEVECTOR_H__
