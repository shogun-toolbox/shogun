/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef __DATATYPE_H__
#define __DATATYPE_H__

#include "lib/common.h"

#define PT_NOT_GENERIC	PT_SGOBJECT
#define PT_LONGEST		floatmax_t

namespace shogun
{

typedef int32_t index_t;

template<class T> class SGVector
{
	public:
		/** default constructor */
		SGVector() : vector(NULL), vlen(0), do_free(false) { }

		/** constructor for setting params */
		SGVector(T* v, index_t len, bool free_vec=false)
			: vector(v), vlen(len), do_free(free_vec) { }

		/** copy constructor */
		SGVector(const SGVector &orig)
			: vector(orig.vector), vlen(orig.vlen) { }

		void free_vector()
		{
			if (do_free)
				delete[] vector;

			vector=NULL;
			do_free=false;
			vlen=0;
		}

	public:
		/** vector  */
		T* vector;
		/** length of vector  */
		index_t vlen;
		/** whether vector needs to be freed */
		bool do_free;
};

template<class T> class SGMatrix
{
	public:
		/** default constructor */
		SGMatrix() : matrix(NULL), num_rows(0), num_cols(0), do_free(false) { }

		/** constructor for setting params */
		SGMatrix(T* m, index_t nrows, index_t ncols, bool free_mat=false)
			: matrix(m), num_rows(nrows), num_cols(ncols), do_free(free_mat) { }

		/** copy constructor */
		SGMatrix(const SGMatrix &orig)
			: matrix(orig.matrix), num_rows(orig.num_rows),
			num_cols(orig.num_cols), do_free(orig.do_free) { }

		void free_matrix()
		{
			if (do_free)
				delete[] matrix;

			matrix=NULL;
			do_free=false;
			num_rows=0;
			num_cols=0;
		}

	public:
		/** matrix  */
		T* matrix;
		/** number of rows of matrix  */
		index_t num_rows;
		/** number of columns of matrix  */
		index_t num_cols;
		/** whether matrix needs to be freed */
		bool do_free;
};

template<class T> struct SGNDArray
{
	/** default constructor */
	SGNDArray() : array(NULL), dims(NULL), num_dims(0) { }

	/** constructor for setting params */
	SGNDArray(T* a, index_t* d, index_t nd)
		: array(a), dims(d), num_dims(nd) { }

	/** copy constructor */
	SGNDArray(const SGNDArray &orig)
	: array(orig.array), dims(orig.dims), num_dims(orig.num_dims) { }

	/** array  */
	T* array;
	/** dimension sizes */
	index_t* dims;
	/** number of dimensions  */
	index_t num_dims;
};

template<class T> struct SGString
{
	/** string  */
	T* string;
	/** length of string  */
	index_t length;
};

/** template class SGStringList */
template <class T> struct SGStringList
{
	/* number of strings */
	int32_t num_strings;

	/** length of longest string */
	int32_t max_string_length;

	/// this contains the array of features.
	SGString<T>* strings;
};

/** template class SGSparseVectorEntry */
template <class T> struct SGSparseVectorEntry
{
	/** feature index  */
	index_t feat_index;
	/** entry ... */
	T entry;
};

/** template class SGSparseVector */
template <class T> struct SGSparseVector
{
	/** vector index */
	index_t vec_index;
	/** number of feature entries */
	index_t num_feat_entries;
	/** features */
	SGSparseVectorEntry<T>* features;
};

/** template class SGSparseMatrix */
template <class T> struct SGSparseMatrix
{
	/// total number of vectors
	int32_t num_vectors;

	/// total number of features
	int32_t num_features;

	/// array of sparse vectors of size num_vectors
	SGSparseVector<T>* sparse_matrix;
};

enum EContainerType
{
	CT_SCALAR,
	CT_VECTOR,
	CT_MATRIX,
	CT_NDARRAY
};

enum EStructType
{
	ST_NONE,
	ST_STRING,
	ST_SPARSE
};

enum EPrimitiveType
{
	PT_BOOL,
	PT_CHAR,
	PT_INT8,
	PT_UINT8,
	PT_INT16,
	PT_UINT16,
	PT_INT32,
	PT_UINT32,
	PT_INT64,
	PT_UINT64,
	PT_FLOAT32,
	PT_FLOAT64,
	PT_FLOATMAX,
	PT_SGOBJECT
};

/* Datatypes that shogun supports. */
struct TSGDataType
{
	EContainerType m_ctype;
	EStructType m_stype;
	EPrimitiveType m_ptype;
	index_t *m_length_y, *m_length_x;

	explicit TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype);
	explicit TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype, index_t* length);
	explicit TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype, index_t* length_y,
						 index_t* length_x);

	bool operator==(const TSGDataType& a);
	inline bool operator!=(const TSGDataType& a)
	{
		return !(*this == a);
	}

	void to_string(char* dest, size_t n) const;
	size_t sizeof_stype(void) const;
	size_t sizeof_ptype(void) const;

	static size_t sizeof_sparseentry(EPrimitiveType ptype);
	static size_t offset_sparseentry(EPrimitiveType ptype);

	static void stype_to_string(char* dest, EStructType stype,
								EPrimitiveType ptype, size_t n);
	static void ptype_to_string(char* dest, EPrimitiveType ptype,
								size_t n);
	static bool string_to_ptype(EPrimitiveType* ptype,
								const char* str);

	/**
	 * @return size of type in bytes
	 */
	size_t get_size();

	/**
	 * @return number of (matrix, vector, scalar) elements of type
	 */
	index_t get_num_elements();
};
}
#endif /* __DATATYPE_H__  */
