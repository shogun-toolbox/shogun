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

#include <shogun/lib/common.h>

#define PT_NOT_GENERIC	PT_SGOBJECT
#define PT_LONGEST		floatmax_t

namespace shogun
{

template<class T> class CCache;

typedef int32_t index_t;

template<class T> class SGVector
{
	public:
		/** default constructor */
		SGVector() : vector(NULL), vlen(0), do_free(false) { }

		/** constructor for setting params */
		SGVector(T* v, index_t len, bool free_vec=false)
			: vector(v), vlen(len), do_free(free_vec) { }

		/** constructor to create new vector in memory */
		SGVector(index_t len, bool free_vec=false)
			: vlen(len), do_free(free_vec)
		{
			vector=SG_MALLOC(T, len);
		}

		/** copy constructor */
		SGVector(const SGVector &orig)
			: vector(orig.vector), vlen(orig.vlen), do_free(orig.do_free) { }

		static SGVector get_vector(SGVector &src, bool own=true)
		{
			if (!own)
				return src;

			src.do_free=false;
			return SGVector(src.vector, src.vlen);
		}

		virtual void free_vector()
		{
			if (do_free)
				SG_FREE(vector);

			vector=NULL;
			do_free=false;
			vlen=0;
		}

		virtual void destroy_vector()
		{
			do_free=true;
			free_vector();
		}

	public:
		/** vector  */
		T* vector;
		/** length of vector  */
		index_t vlen;
		/** whether vector needs to be freed */
		bool do_free;
};

template<class T> class SGCachedVector : public SGVector<T>
{
	public:
		/** default constructor */
		SGCachedVector(CCache<T>* c, int32_t i)
			: SGVector<T>(), cache(c), idx(i)
		{
		}

		/** constructor for setting params */
		SGCachedVector(CCache<T>* c, int32_t i,
				T* v, index_t len, bool free_vec=false)
			: SGVector<T>(v, len, free_vec), cache(c), idx(i)
		{
		}

		/** constructor to create new vector in memory */
		SGCachedVector(CCache<T>* c, int32_t i, index_t len, bool free_vec=false) :
			SGVector<T>(len, free_vec), cache(c), idx(i)
		{
		}
 
		virtual void free_vector()
		{
			//clean up cache fixme
			SGVector<T>::free_vector();
		}

		virtual void destroy_vector()
		{
			//clean up cache fixme
			SGVector<T>::destroy_vector();
			if (cache)
				cache->unlock_entry(idx);
		}

	public:	
		int idx;
		CCache<T>* cache;
};

template<class T> class SGMatrix
{
	public:
		/** default constructor */
		SGMatrix() : matrix(NULL), num_rows(0), num_cols(0), do_free(false) { }

		/** constructor for setting params */
		SGMatrix(T* m, index_t nrows, index_t ncols, bool free_mat=false)
			: matrix(m), num_rows(nrows), num_cols(ncols), do_free(free_mat) { }

		/** constructor to create new matrix in memory */
		SGMatrix(index_t nrows, index_t ncols, bool free_mat=false)
			: num_rows(nrows), num_cols(ncols), do_free(free_mat)
		{
			matrix=SG_MALLOC(T, nrows*ncols);
		}

		/** copy constructor */
		SGMatrix(const SGMatrix &orig)
			: matrix(orig.matrix), num_rows(orig.num_rows),
			num_cols(orig.num_cols), do_free(orig.do_free) { }

		void free_matrix()
		{
			if (do_free)
				SG_FREE(matrix);

			matrix=NULL;
			do_free=false;
			num_rows=0;
			num_cols=0;
		}

		void destroy_matrix()
		{
			do_free=true;
			free_matrix();
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

template<class T> class SGNDArray
{
    public:
        /** default constructor */
        SGNDArray() : array(NULL), dims(NULL), num_dims(0) { }

        /** constructor for setting params */
        SGNDArray(T* a, index_t* d, index_t nd)
            : array(a), dims(d), num_dims(nd) { }

        /** copy constructor */
        SGNDArray(const SGNDArray &orig)
            : array(orig.array), dims(orig.dims), num_dims(orig.num_dims) { }

    public:
        /** array  */
        T* array;
        /** dimension sizes */
        index_t* dims;
        /** number of dimensions  */
        index_t num_dims;
};

template<class T> struct SGString
{
public:
	/** default constructor */
	SGString() : string(NULL), slen(0), do_free(false) { }

	/** constructor for setting params */
	SGString(T* s, index_t l, bool free=false)
		: string(s), slen(l), do_free(free) { }

	/** constructor for setting params from a SGVector*/
	SGString(SGVector<T> v)
		: string(v.vector), slen(v.vlen), do_free(v.do_free) { }

	/** constructor to create new string in memory */
	SGString(index_t len, bool free=false) :
		slen(len), do_free(free)
	{
		string=SG_MALLOC(T, len);
	}

	/** copy constructor */
	SGString(const SGString &orig)
		: string(orig.string), slen(orig.slen), do_free(orig.do_free) { }

	void free_string()
	{
		if (do_free)
			SG_FREE(string);

		string=NULL;
		do_free=false;
		slen=0;
	}

	void destroy_string()
	{
		do_free=true;
		free_string();
	}

public:
	/** string  */
	T* string;
	/** length of string  */
	index_t slen;
	/** whether string needs to be freed */
	bool do_free;
};

/** template class SGStringList */
template <class T> struct SGStringList
{
public:
	/** default constructor */
	SGStringList() : strings(NULL), num_strings(0), max_string_length(0),
		do_free(false) { }

	/** constructor for setting params */
	SGStringList(SGString<T>* s, index_t num_s, index_t max_length,
			bool free_strings=false) : num_strings(num_s),
			max_string_length(max_length), strings(s), do_free(free_strings) { }

	/** constructor to create new string list in memory */
	SGStringList(index_t num_s, index_t max_length, bool free_strings=false)
		: num_strings(num_s), max_string_length(max_length),
		  do_free(free_strings)
	{
		strings=SG_MALLOC(SGString<T>, num_strings);
	}

	/** copy constructor */
	SGStringList(const SGStringList &orig) :
		num_strings(orig.num_strings),
		max_string_length(orig.max_string_length),
		strings(orig.strings), do_free(orig.do_free) { }

	void free_list()
	{
		if (do_free)
			SG_FREE(strings);

		strings=NULL;
		do_free=false;
		num_strings=0;
		max_string_length=0;
	}

	void destroy_list()
	{
		do_free=true;
		free_list();
	}

public:
	/* number of strings */
	index_t num_strings;

	/** length of longest string */
	index_t max_string_length;

	/// this contains the array of features.
	SGString<T>* strings;

	/** whether vector needs to be freed */
	bool do_free;
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
public:
	/** default constructor */
	SGSparseVector() :
		vec_index(0), num_feat_entries(0), features(NULL), do_free(false) {}

	/** constructor for setting params */
	SGSparseVector(SGSparseVectorEntry<T>* feats, index_t num_entries,
			index_t index, bool free=false) :
			vec_index(index), num_feat_entries(num_entries), features(feats),
			do_free(free) {}

	/** constructor to create new vector in memory */
	SGSparseVector(index_t num_entries, index_t index, bool free=false) :
		vec_index(index), num_feat_entries(num_entries), do_free(false)
	{
		features=SG_MALLOC(SGSparseVectorEntry<T>, num_feat_entries);
	}

	/** copy constructor */
	SGSparseVector(const SGSparseVector& orig) :
			vec_index(orig.vec_index), num_feat_entries(orig.num_feat_entries),
			features(orig.features), do_free(orig.do_free) {}

	void free_vector()
	{
		if (do_free)
			SG_FREE(features);

		features=NULL;
		do_free=false;
		vec_index=0;
		num_feat_entries=0;
	}

	void destroy_vector()
	{
		do_free=true;
		free_vector();
	}

public:
	/** vector index */
	index_t vec_index;

	/** number of feature entries */
	index_t num_feat_entries;

	/** features */
	SGSparseVectorEntry<T>* features;

	/** whether vector needs to be freed */
	bool do_free;
};

/** template class SGSparseMatrix */
template <class T> class SGSparseMatrix
{
	public:
		/** default constructor */
		SGSparseMatrix() :
			num_vectors(0), num_features(0), sparse_matrix(NULL),
			do_free(false) { }


		/** constructor for setting params */
		SGSparseMatrix(SGSparseVector<T>* vecs, index_t num_feat,
				index_t num_vec, bool free=false) :
			num_vectors(num_vec), num_features(num_feat),
			sparse_matrix(vecs), do_free(free) { }

		/** constructor to create new matrix in memory */
		SGSparseMatrix(index_t num_vec, index_t num_feat, bool free=false) :
			num_vectors(num_vectors), num_features(num_feat), do_free(free)
		{
			sparse_matrix=SG_MALLOC(SGSparseVector<T>, num_vectors);
		}

		/** copy constructor */
		SGSparseMatrix(const SGSparseMatrix &orig) :
			num_vectors(orig.num_vectors), num_features(orig.num_features),
			sparse_matrix(orig.sparse_matrix), do_free(orig.do_free) { }

		void free_matrix()
		{
			if (do_free)
				SG_FREE(sparse_matrix);

			sparse_matrix=NULL;
			do_free=false;
			num_vectors=0;
			num_features=0;
		}

		void destroy_matrix()
		{
			do_free=true;
			free_matrix();
		}

	public:
	/// total number of vectors
	int32_t num_vectors;

	/// total number of features
	int32_t num_features;

	/// array of sparse vectors of size num_vectors
	SGSparseVector<T>* sparse_matrix;

	/** whether vector needs to be freed */
	bool do_free;
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
