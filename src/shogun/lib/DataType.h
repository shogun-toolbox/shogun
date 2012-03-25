/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Wrriten (W) 2012 Fernando José Iglesias García
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef __DATATYPE_H__
#define __DATATYPE_H__

#include <shogun/lib/common.h>
//#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>

#define PT_NOT_GENERIC	PT_SGOBJECT
#define PT_LONGEST	floatmax_t

namespace shogun
{

//class CMath;
template<class T> class CCache;

/** index */
typedef int32_t index_t;

/** @brief shogun vector */
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

		/** empty destructor */
		virtual ~SGVector()
		{
		}

		/** get vector
		 * @param src vector to get
		 * @param own true if should be owned
		 */
		static SGVector get_vector(SGVector &src, bool own=true)
		{
			if (!own)
				return src;

			src.do_free=false;
			return SGVector(src.vector, src.vlen);
		}

		/** fill vector with zeros */
		void zero()
		{
			if (vector && vlen)
				set_const(0);
		}

		/** set vector to a constant */
		void set_const(T const_elem)
		{
			for (index_t i=0; i<vlen; i++)
				vector[i]=const_elem ;
		}

		/** range fill */
		void range_fill(T start=0)
		{
			range_fill_vector(vector, vlen, start);
		}

		/** random */
		void random(T min_value, T max_value)
		{
			random_vector(vector, vlen, min_value, max_value);
		}

		/** random permutate */
		void randperm()
		{
			/* this does not work. Heiko Strathmann */
			SG_SNOTIMPLEMENTED;
			randperm(vector, vlen);
		}

		/** clone vector */
		SGVector<T> clone()
		{
			SGVector<T> c;
			c.vector=clone_vector(vector, vlen);
			c.vlen=vlen;
			c.do_free=true;

			return c;
		}

		/** clone vector */
		template <class VT>
		static VT* clone_vector(const VT* vec, int32_t len)
		{
			VT* result = SG_MALLOC(VT, len);
			for (int32_t i=0; i<len; i++)
				result[i]=vec[i];

			return result;
		}

		/** fill vector */
		template <class VT>
		static void fill_vector(VT* vec, int32_t len, VT value)
		{
			for (int32_t i=0; i<len; i++)
				vec[i]=value;
		}

		/** range fill vector */
		template <class VT>
		static void range_fill_vector(VT* vec, int32_t len, VT start=0)
		{
			for (int32_t i=0; i<len; i++)
				vec[i]=i+start;
		}

		/** random vector */
		template <class VT>
		static void random_vector(VT* vec, int32_t len, VT min_value, VT max_value)
		{
			//FIXME for (int32_t i=0; i<len; i++)
			//FIXME 	vec[i]=CMath::random(min_value, max_value);
		}

		/** random permatutaion */
		template <class VT>
		static void randperm(VT* perm, int32_t n)
		{
			for (int32_t i = 0; i < n; i++)
				perm[i] = i;
			permute(perm,n);
		}

		/** permute */
		template <class VT>
		static void permute(VT* perm, int32_t n)
		{
			//FIXME for (int32_t i = 0; i < n; i++)
			//FIXME 	CMath::swap(perm[random(0, n - 1)], perm[i]);
		}

		/** get vector element at index
		 *
		 * @param index index
		 * @return vector element at index
		 */
		const T& get_element(index_t index)
		{
			ASSERT(vector && (index>=0) && (index<vlen));
			return vector[index];
		}

		/** set vector element at index 'index' return false in case of trouble
		 *
		 * @param p_element vector element to set
		 * @param index index
		 * @return if setting was successful
		 */
		void set_element(const T& p_element, index_t index)
		{
			ASSERT(vector && (index>=0) && (index<vlen));
			vector[index]=p_element;
		}

		/** resize vector
		 *
		 * @param n new size
		 * @return if resizing was successful
		 */
		void resize_vector(int32_t n)
		{
			vector=SG_REALLOC(T, vector, n);

			if (n > vlen)
				memset(&vector[vlen], 0, (n-vlen)*sizeof(T));
			vlen=n;
		}

		/** operator overload for vector read only access
		 *
		 * @param index dimension to access
		 *
		 */
		inline const T& operator[](index_t index) const
		{
			return vector[index];
		}

		/** operator overload for vector r/w access
		 *
		 * @param index dimension to access
		 *
		 */
		inline T& operator[](index_t index)
		{
			return vector[index];
		}

		/** free vector */
		virtual void free_vector()
		{
			if (do_free)
				SG_FREE(vector);

			vector=NULL;
			do_free=false;
			vlen=0;
		}

		/** destroy vector */
		virtual void destroy_vector()
		{
			do_free=true;
			free_vector();
		}

		/** display array size */
		void display_size() const
		{
			SG_SPRINT("SGVector '%p' of size: %d\n", vector, vlen);
		}

		/** display array */
		void display_vector() const
		{
			display_size();
			for (int32_t i=0; i<vlen; i++)
				SG_SPRINT("%10.10g,", (float64_t) vector[i]);
			SG_SPRINT("\n");
		}

	public:
		/** vector  */
		T* vector;
		/** length of vector  */
		index_t vlen;
		/** whether vector needs to be freed */
		bool do_free;
};

//template<class T> class SGCachedVector : public SGVector<T>
//{
//	public:
//		/** default constructor */
//		SGCachedVector(CCache<T>* c, index_t i)
//			: SGVector<T>(), cache(c), idx(i)
//		{
//		}
//
//		/** constructor for setting params */
//		SGCachedVector(CCache<T>* c, index_t i,
//				T* v, index_t len, bool free_vec=false)
//			: SGVector<T>(v, len, free_vec), cache(c), idx(i)
//		{
//		}
//
//		/** constructor to create new vector in memory */
//		SGCachedVector(CCache<T>* c, index_t i, index_t len, bool free_vec=false) :
//			SGVector<T>(len, free_vec), cache(c), idx(i)
//		{
//		}
//
// 		/** free vector */
//		virtual void free_vector()
//		{
//			//clean up cache fixme
//			SGVector<T>::free_vector();
//		}
//
//		/** destroy vector */
//		virtual void destroy_vector()
//		{
//			//clean up cache fixme
//			SGVector<T>::destroy_vector();
//			if (cache)
//				cache->unlock_entry(idx);
//		}
//
//	public:
//		/** idx */
//		index_t idx;
//
//		/** cache */
//		CCache<T>* cache;
//};

/** @brief shogun matrix */
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

		/** empty destructor */
		virtual ~SGMatrix()
		{
		}

		/** free matrix */
		virtual void free_matrix()
		{
			if (do_free)
				SG_FREE(matrix);

			matrix=NULL;
			do_free=false;
			num_rows=0;
			num_cols=0;
		}

		/** destroy matrix */
		virtual void destroy_matrix()
		{
			do_free=true;
			free_matrix();
		}

		/** get a column vector
		 * @param col column index
		 */
		T* get_column_vector(index_t col) const
		{
			return &matrix[col*num_rows];
		}

		/** operator overload for matrix read only access
		 * @param index to access
		 */
		inline const T& operator[](index_t index) const
		{
			return matrix[index];
		}

		/** operator overload for matrix r/w access
		 * @param index to access
		 */
		inline T& operator[](index_t index)
		{
			return matrix[index];
		}

		/** set matrix to a constant */
		void set_const(T const_elem)
		{
			for (index_t i=0; i<num_rows*num_cols; i++)
				matrix[i]=const_elem ;
		}

		/** fill matrix with zeros */
		void zero()
		{
			if (matrix && (num_rows*num_cols))
				set_const(0);
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

/** @brief shogun n-dimensional array */
template<class T> class SGNDArray
{
    public:
        /** default constructor */
        SGNDArray() : array(NULL), dims(NULL), num_dims(0) { }

        /** constructor for setting params */
        SGNDArray(T* a, index_t* d, index_t nd)
            : array(a), dims(d), num_dims(nd) { }

	/** constructor to create new ndarray in memory */
	SGNDArray(index_t* d, index_t nd)
		: dims(d), num_dims(nd)
	{
		index_t tot = 1;
		for (int32_t i=0; i<nd; i++)
			tot *= dims[i];
		array=SG_MALLOC(T, tot);
	}

        /** copy constructor */
        SGNDArray(const SGNDArray &orig)
            : array(orig.array), dims(orig.dims), num_dims(orig.num_dims) { }

	/** empty destructor */
	virtual ~SGNDArray()
	{
	}

	/** destroy ndarry */
	virtual void destroy_ndarray()
	{
		SG_FREE(array);
		SG_FREE(dims);

		array     = NULL;
		dims      = NULL;
		num_dims  = 0;
	}

	/** get a matrix formed by the two first dimensions
	 *
	 * @param  matIdx matrix index
	 * @return pointer to the matrix
	 */
	T* get_matrix(index_t matIdx) const
	{	
		ASSERT(array && dims && num_dims > 2 && dims[2] > matIdx);
		return &array[matIdx*dims[0]*dims[1]];
	}
	
	/** operator overload for ndarray read only access
	 *
	 * @param index to access
	 */
	inline const T& operator[](index_t index) const
	{
		return array[index];
	}

	/** operator overload for ndarray r/w access
	 *
	 * @param index to access
	 */
	inline T& operator[](index_t index)
	{
		return array[index];
	}

	/** transposes a matrix formed by the two first dimensions
	 *
	 * @param matIdx matrix index
	 */
	void transpose_matrix(index_t matIdx) const
	{
		ASSERT(array && dims && num_dims > 2 && dims[2] > matIdx);
		
		T aux;
		// Index to acces directly the elements of the matrix of interest
		int32_t idx = matIdx*dims[0]*dims[1];

		for (int32_t i=0; i<dims[0]; i++)
			for (int32_t j=0; j<i-1; j++)
			{
				aux = array[idx + i + j*dims[0]];
				array[idx + i + j*dims[0]] = array[idx + j + i*dims[0]];
				array[idx + j + i*dims[1]] = aux;
			}

		// Swap the sizes of the two first dimensions
		index_t auxDim = dims[0];
		dims[0] = dims[1];
		dims[1] = auxDim;
	}

    public:
        /** array  */
        T* array;
        /** dimension sizes */
        index_t* dims;
        /** number of dimensions  */
        index_t num_dims;
};

/** @brief shogun string */
template<class T> class SGString
{
public:
	/** default constructor */
	SGString() : string(NULL), slen(0), do_free(false) { }

	/** constructor for setting params */
	SGString(T* s, index_t l, bool free_s=false)
		: string(s), slen(l), do_free(free_s) { }

	/** constructor for setting params from a SGVector*/
	SGString(SGVector<T> v)
		: string(v.vector), slen(v.vlen), do_free(v.do_free) { }

	/** constructor to create new string in memory */
	SGString(index_t len, bool free_s=false) :
		slen(len), do_free(free_s)
	{
		string=SG_MALLOC(T, len);
	}

	/** copy constructor */
	SGString(const SGString &orig)
		: string(orig.string), slen(orig.slen), do_free(orig.do_free) { }

	/** free string */
	void free_string()
	{
		if (do_free)
			SG_FREE(string);

		string=NULL;
		do_free=false;
		slen=0;
	}

	/** destroy string */
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

/** @brief template class SGStringList */
template <class T> struct SGStringList
{
public:
	/** default constructor */
	SGStringList() : num_strings(0), max_string_length(0), strings(NULL), 
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

	/** free list */
	void free_list()
	{
		if (do_free)
			SG_FREE(strings);

		strings=NULL;
		do_free=false;
		num_strings=0;
		max_string_length=0;
	}

	/** destroy list */
	void destroy_list()
	{
		do_free=true;
		free_list();
	}

public:
	/** number of strings */
	index_t num_strings;

	/** length of longest string */
	index_t max_string_length;

	/** this contains the array of features */
	SGString<T>* strings;

	/** whether vector needs to be freed */
	bool do_free;
};

/** @brief template class SGSparseVectorEntry */
template <class T> struct SGSparseVectorEntry
{
	/** feature index  */
	index_t feat_index;
	/** entry ... */
	T entry;
};

/** @brief template class SGSparseVector */
template <class T> class SGSparseVector
{
public:
	/** default constructor */
	SGSparseVector() :
		vec_index(0), num_feat_entries(0), features(NULL), do_free(false) {}

	/** constructor for setting params */
	SGSparseVector(SGSparseVectorEntry<T>* feats, index_t num_entries,
			index_t index, bool free_v=false) :
			vec_index(index), num_feat_entries(num_entries), features(feats),
			do_free(free_v) {}

	/** constructor to create new vector in memory */
	SGSparseVector(index_t num_entries, index_t index, bool free_v=false) :
		vec_index(index), num_feat_entries(num_entries), do_free(free_v)
	{
		features=SG_MALLOC(SGSparseVectorEntry<T>, num_feat_entries);
	}

	/** copy constructor */
	SGSparseVector(const SGSparseVector& orig) :
			vec_index(orig.vec_index), num_feat_entries(orig.num_feat_entries),
			features(orig.features), do_free(orig.do_free) {}

	/** free vector */
	void free_vector()
	{
		if (do_free)
			SG_FREE(features);

		features=NULL;
		do_free=false;
		vec_index=0;
		num_feat_entries=0;
	}

	/** destroy vector */
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

/** @brief template class SGSparseMatrix */
template <class T> class SGSparseMatrix
{
	public:
		/** default constructor */
		SGSparseMatrix() :
			num_vectors(0), num_features(0), sparse_matrix(NULL),
			do_free(false) { }


		/** constructor for setting params */
		SGSparseMatrix(SGSparseVector<T>* vecs, index_t num_feat,
				index_t num_vec, bool free_m=false) :
			num_vectors(num_vec), num_features(num_feat),
			sparse_matrix(vecs), do_free(free_m) { }

		/** constructor to create new matrix in memory */
		SGSparseMatrix(index_t num_vec, index_t num_feat, bool free_m=false) :
			num_vectors(num_vec), num_features(num_feat), do_free(free_m)
		{
			sparse_matrix=SG_MALLOC(SGSparseVector<T>, num_vectors);
		}

		/** copy constructor */
		SGSparseMatrix(const SGSparseMatrix &orig) :
			num_vectors(orig.num_vectors), num_features(orig.num_features),
			sparse_matrix(orig.sparse_matrix), do_free(orig.do_free) { }

		/** free matrix */
		void free_matrix()
		{
			if (do_free)
				SG_FREE(sparse_matrix);

			sparse_matrix=NULL;
			do_free=false;
			num_vectors=0;
			num_features=0;
		}

		/** own matrix */
		void own_matrix()
		{
			for (index_t i=0; i<num_vectors; i++)
				sparse_matrix[i].do_free=false;

			do_free=false;
		}

		/** destroy matrix */
		void destroy_matrix()
		{
			do_free=true;
			free_matrix();
		}

	public:
	/// total number of vectors
	index_t num_vectors;

	/// total number of features
	index_t num_features;

	/// array of sparse vectors of size num_vectors
	SGSparseVector<T>* sparse_matrix;

	/** whether vector needs to be freed */
	bool do_free;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum EContainerType
{
	CT_SCALAR=0,
	CT_VECTOR=1,
	CT_MATRIX=2,
	CT_NDARRAY=3,
	CT_SGVECTOR=4,
	CT_SGMATRIX=5
};

enum EStructType
{
	ST_NONE=0,
	ST_STRING=1,
	ST_SPARSE=2
};

enum EPrimitiveType
{
	PT_BOOL=0,
	PT_CHAR=1,
	PT_INT8=2,
	PT_UINT8=3,
	PT_INT16=4,
	PT_UINT16=5,
	PT_INT32=6,
	PT_UINT32=7,
	PT_INT64=8,
	PT_UINT64=9,
	PT_FLOAT32=10,
	PT_FLOAT64=11,
	PT_FLOATMAX=12,
	PT_SGOBJECT=13
};
#endif

/** @brief Datatypes that shogun supports. */
struct TSGDataType
{
	/** container type */
	EContainerType m_ctype;
	/** struct type */
	EStructType m_stype;
	/** primitive type */
	EPrimitiveType m_ptype;

	/** length y */
	index_t *m_length_y;
	/** length x */
	index_t *m_length_x;

	/** constructor 
	 * @param ctype
	 * @param stype
	 * @param ptype
	 */
	explicit TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype);
	/** constructor
	 * @param ctype
	 * @param stype
	 * @param ptype
	 * @param length
	 */
	explicit TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype, index_t* length);
	/** constructor
	 * @param ctype
	 * @param stype
	 * @param ptype
	 * @param length_y
	 * @param length_x
	 */
	explicit TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype, index_t* length_y,
						 index_t* length_x);

	/** equality */
	bool operator==(const TSGDataType& a);
	/** inequality
	 * @param a
	 */
	inline bool operator!=(const TSGDataType& a)
	{
		return !(*this == a);
	}

	/** to string
	 * @param dest
	 * @param n
	 */
	void to_string(char* dest, size_t n) const;

	/** size of stype */
	size_t sizeof_stype() const;
	/** size of ptype */
	size_t sizeof_ptype() const;

	/** size of sparse entry 
	 * @param ptype
	 */
	static size_t sizeof_sparseentry(EPrimitiveType ptype);

	/** offset of sparse entry
	 * @param ptype
	 */
	static size_t offset_sparseentry(EPrimitiveType ptype);

	/** stype to string
	 * @param dest
	 * @param stype
	 * @param ptype
	 * @param n
	 */
	static void stype_to_string(char* dest, EStructType stype,
	                            EPrimitiveType ptype, size_t n);
	/** ptype to string 
	 * @param dest
	 * @param ptype
	 * @param n
	 */
	static void ptype_to_string(char* dest, EPrimitiveType ptype,
	                            size_t n);
	/** string to ptype 
	 * @param ptype
	 * @param str
	 */
	static bool string_to_ptype(EPrimitiveType* ptype,
	                            const char* str);

	/** get size
	 * @return size of type in bytes
	 */
	size_t get_size();

	/** get num of elements
	 * @return number of (matrix, vector, scalar) elements of type
	 */
	index_t get_num_elements();
};
}
#endif /* __DATATYPE_H__  */
