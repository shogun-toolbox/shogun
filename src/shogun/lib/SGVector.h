/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Heiko Strathmann
 * Written (W) 2013 Soumyajit De
 * Written (W) 2012 Fernando Jose Iglesias Garcia
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */
#ifndef __SGVECTOR_H__
#define __SGVECTOR_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/lib/SGReferencedData.h>

namespace Eigen
{
	template <class, int, int, int, int, int> class Matrix;
	template<int, int> class Stride;
	template <class, int, class> class Map;
}

namespace shogun
{
	template <class T> class SGSparseVector;
	template <class T> class SGMatrix;
	class CFile;
	class CRandom;

/** @brief shogun vector */
template<class T> class SGVector : public SGReferencedData
{
	typedef Eigen::Matrix<T,-1,1,0,-1,1> EigenVectorXt;
	typedef Eigen::Matrix<T,1,-1,0x1,1,-1> EigenRowVectorXt;

	typedef Eigen::Map<EigenVectorXt,0,Eigen::Stride<0,0> > EigenVectorXtMap;
	typedef Eigen::Map<EigenRowVectorXt,0,Eigen::Stride<0,0> > EigenRowVectorXtMap;

	public:
		/** The scalar type of the vector */
		typedef T Scalar;

		/** Default constructor */
		SGVector();

		/** Constructor for setting params */
		SGVector(T* v, index_t len, bool ref_counting=true);

		/** Wraps a vector around an existing memory segment with an offset */
		SGVector(T* m, index_t len, index_t offset)
			: SGReferencedData(false), vector(m+offset), vlen(len) { }

		/** Constructor to create new vector in memory */
		SGVector(index_t len, bool ref_counting=true);

		/** Copy constructor */
		SGVector(const SGVector &orig);

#ifndef SWIG // SWIG should skip this part
#ifdef HAVE_EIGEN3
		/** Wraps a matrix around the data of an Eigen3 column vector */
		SGVector(EigenVectorXt& vec);

		/** Wraps a matrix around the data of an Eigen3 row vector */
		SGVector(EigenRowVectorXt& vec);

		/** Wraps an Eigen3 column vector around the data of this matrix */
		operator EigenVectorXtMap() const;

		/** Wraps an Eigen3 row vector around the data of this matrix */
		operator EigenRowVectorXtMap() const;
#endif
#endif

		/** Set vector to a constant
		 *
		 * @param const_elem - value to set vector to
		 */
		void set_const(T const_elem);

		/**
		 * Get the vector (no copying is done here)
		 *
		 * @return the refcount increased vector
		 */
		SGVector<T> get()
		{
			return *this;
		}

#ifndef SWIG // SWIG should skip this part
		/** Wrapper for the copy constructor useful for SWIG interfaces
		 *
		 * @param orig vector to set
		 */
		void set(SGVector<T> orig);

		/** Empty destructor */
		virtual ~SGVector();

		/** Size */
		inline int32_t size() const { return vlen; }

		/** Cast to pointer */
		operator T*() { return vector; };

		/** Fill vector with zeros */
		void zero();

		/** Range fill a vector with start...start+len-1
		 *
		 * @param start - value to be assigned to first element of vector
		 */
		void range_fill(T start=0);

		/** Create random vector
		 *
		 * @param min_value [min_value,max_value]
		 * @param max_value
		 */
		void random(T min_value, T max_value);

		/** For a sorted (ascending) vector, gets the index after the first
		 * element that is smaller than the given one
		 *
		 * @param element element to find index for
		 * @return index of the first element greater than given one
		 */
		index_t find_position_to_insert(T element);

		/** Quicksort the vector
		 * it is sorted from in ascending (for type T)
		 */
		void qsort();

		/** Get sorted index.
		 *
		 * idx = v.argsort() is similar to Matlab [~, idx] = sort(v)
		 *
		 * @return sorted index for this vector
		 */
		SGVector<index_t> argsort();

		/** Check if vector is sorted
		 *
		 * @return true if vector is sorted, false otherwise
		 */
		bool is_sorted() const;

		/** Clone vector */
		SGVector<T> clone() const;

		/** Clone vector */
		static T* clone_vector(const T* vec, int32_t len);

		/** Fill vector */
		static void fill_vector(T* vec, int32_t len, T value);

		/** Range fill vector */
		static void range_fill_vector(T* vec, int32_t len, T start=0);

		/** Random vector */
		static void random_vector(T* vec, int32_t len, T min_value, T max_value);

		/** Get vector element at index
		 *
		 * @param index index
		 * @return vector element at index
		 */
		const T& get_element(index_t index);

		/** Set vector element at index 'index' return false in case of trouble
		 *
		 * @param p_element vector element to set
		 * @param index index
		 * @return if setting was successful
		 */
		void set_element(const T& p_element, index_t index);

		/** Resize vector
		 *
		 * @param n new size
		 * @return if resizing was successful
		 */
		void resize_vector(int32_t n);

		/** Operator overload for vector read only access
		 *
		 * @param index dimension to access
		 *
		 */
		inline const T& operator[](uint64_t index) const
		{
			return vector[index];
		}

		/** Operator overload for vector read only access
		 *
		 * @param index dimension to access
		 *
		 */
		inline const T& operator[](int64_t index) const
		{
			return vector[index];
		}

		/** Operator overload for vector read only access
		 *
		 * @param index dimension to access
		 *
		 */
		inline const T& operator[](uint32_t index) const
		{
			return vector[index];
		}

		/** Operator overload for vector read only access
		 *
		 * @param index dimension to access
		 *
		 */
		inline const T& operator[](int32_t index) const
		{
			return vector[index];
		}

		/** Operator overload for vector r/w access
		 *
		 * @param index dimension to access
		 *
		 */
		inline T& operator[](uint64_t index)
		{
			return vector[index];
		}

		/** Operator overload for vector r/w access
		 *
		 * @param index dimension to access
		 *
		 */
		inline T& operator[](int64_t index)
		{
			return vector[index];
		}

		/** Operator overload for vector r/w access
		 *
		 * @param index dimension to access
		 *
		 */
		inline T& operator[](uint32_t index)
		{
			return vector[index];
		}

		/** Operator overload for vector r/w access
		 *
		 * @param index dimension to access
		 *
		 */
		inline T& operator[](int32_t index)
		{
			return vector[index];
		}

		/** Add vector to current vector
		 *
		 * @param x add vector x to current vector
		 */
		void add(const SGVector<T> x);

		/** Add sparse vector to current vector
		 *
		 * @param x add sparse vector x to current vector
		 */
		void add(const SGSparseVector<T>& x);

		/** Add scalar to current vector
		 *
		 * @param x add vector x to current vector
		 */
		void add(const T x);

		/** Addition operator */
		SGVector<T> operator+ (SGVector<T> x);

		/** Inplace addition operator */
		SGVector<T> operator+= (SGVector<T> x)
		{
			add(x);
			return *this;
		}

		/** Inplace addition operator for sparse vector */
		SGVector<T> operator+= (SGSparseVector<T>& x)
		{
			add(x);
			return *this;
		}

		/** Equals method up to precision for vectors (element-wise)
		 * @param other vector to compare with
		 * @return false if any element differs or if sizes are different,
		 * true otherwise
		 */
		bool equals(SGVector<T>& other);

		/// || x ||_2
		static T twonorm(const T* x, int32_t len);

		/// || x ||_1
		static float64_t onenorm(T* x, int32_t len);

		/// || x ||_q^q
		static T qsq(T* x, int32_t len, float64_t q);

		/// || x ||_q
		static T qnorm(T* x, int32_t len, float64_t q);

		/// x=x+alpha*y
		static void vec1_plus_scalar_times_vec2(T* vec1,
				const T scalar, const T* vec2, int32_t n);

		/// Compute dot product between v1 and v2 (blas optimized)
		static inline float64_t dot(const bool* v1, const bool* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((v1[i]) ? 1 : 0) * ((v2[i]) ? 1 : 0);
			return r;
		}

		/// Compute dot product between v1 and v2 (blas optimized)
		static inline floatmax_t dot(const floatmax_t* v1, const floatmax_t* v2, int32_t n)
		{
			floatmax_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=v1[i]*v2[i];
			return r;
		}


		/// Compute dot product between v1 and v2 (blas optimized)
		static float64_t dot(const float64_t* v1, const float64_t* v2, int32_t n);

		/// Compute dot product between v1 and v2 (blas optimized)
		static float32_t dot(const float32_t* v1, const float32_t* v2, int32_t n);

		/// compute dot product between v1 and v2 (for 64bit unsigned ints)
		static inline float64_t dot(
			const uint64_t* v1, const uint64_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}
		/// Compute dot product between v1 and v2 (for 64bit ints)
		static inline float64_t dot(
			const int64_t* v1, const int64_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// Compute dot product between v1 and v2 (for 32bit ints)
		static inline float64_t dot(
			const int32_t* v1, const int32_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// Compute dot product between v1 and v2 (for 32bit unsigned ints)
		static inline float64_t dot(
			const uint32_t* v1, const uint32_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// Compute dot product between v1 and v2 (for 16bit unsigned ints)
		static inline float64_t dot(
			const uint16_t* v1, const uint16_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// Compute dot product between v1 and v2 (for 16bit unsigned ints)
		static inline float64_t dot(
			const int16_t* v1, const int16_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// Compute dot product between v1 and v2 (for 8bit (un)signed ints)
		static inline float64_t dot(
			const char* v1, const char* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// Compute dot product between v1 and v2 (for 8bit (un)signed ints)
		static inline float64_t dot(
			const uint8_t* v1, const uint8_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// Compute dot product between v1 and v2 (for 8bit (un)signed ints)
		static inline float64_t dot(
			const int8_t* v1, const int8_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// Compute dot product between v1 and v2
		static inline float64_t dot(
			const float64_t* v1, const char* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// Compute vector multiplication
		static inline void vector_multiply(
				T* target, const T* v1, const T* v2,int32_t len)
			{
				for (int32_t i=0; i<len; i++)
					target[i]=v1[i]*v2[i];
			}


		/// target=alpha*vec1 + beta*vec2
		static inline void add(
			T* target, T alpha, const T* v1, T beta, const T* v2,
			int32_t len)
		{
			for (int32_t i=0; i<len; i++)
				target[i]=alpha*v1[i]+beta*v2[i];
		}

		/// Add scalar to vector inplace
		static inline void add_scalar(T alpha, T* vec, int32_t len)
		{
			for (int32_t i=0; i<len; i++)
				vec[i]+=alpha;
		}

		/// Scale vector inplace
		static void scale_vector(T alpha, T* vec, int32_t len);

		/// Return sum(vec)
		static inline T sum(T* vec, int32_t len)
		{
			T result=0;
			for (int32_t i=0; i<len; i++)
				result+=vec[i];

			return result;
		}

		/// Return sum(vec)
		static inline T sum(SGVector<T> vec)
		{
			return sum(vec.vector, vec.vlen);
		}

		/// Return the product of the vectors elements
		static inline T product(T* vec, int32_t len)
		{
			T result=1;
			for (int32_t i=0; i<len; i++)
				result*=vec[i];

			return result;
		}

		/// Return product(vec)
		inline T product()
		{
			return product(vector, vlen);
		}

		/** @return sum(abs(vec)) */
		static T sum_abs(T* vec, int32_t len);

		/** @return sum(abs(vec)) */
		static bool fequal(T x, T y, float64_t precision=1e-6);

		/** Performs a inplace unique of a vector of type T using quicksort
		 * returns the new number of elements
		 */
		static int32_t unique(T* output, int32_t size);

		/** Display array size */
		void display_size() const;

		/** Display vector */
		void display_vector(const char* name="vector",
				const char* prefix="") const;

		/// Display vector (useful for debugging)
		static void display_vector(
			const T* vector, int32_t n, const char* name="vector",
			const char* prefix="");

		/// Display vector (useful for debugging)
		static void display_vector(
			const SGVector<T>, const char* name="vector",
			const char* prefix="");

		/** Find index for occurance of an element
		 * @param elem the element to find
		 */
		SGVector<index_t> find(T elem);

		/** Find index for elements where the predicate returns true
		 * @param p the predicate, it should accept the value of the element and return a bool
		 */
		template <typename Predicate>
		SGVector<index_t> find_if(Predicate p)
		{
			SGVector<index_t> idx(vlen);
			index_t k=0;

			for (index_t i=0; i < vlen; ++i)
				if (p(vector[i]))
					idx[k++] = i;

			idx.vlen = k;
			return idx;
		}

		/// Scale vector inplace
		void scale(T alpha);

		/** Load vector from file
		 *
		 * @param loader File object via which to load data
		 */
		void load(CFile* loader);

		/** Save vector to file
		 *
		 * @param saver File object via which to save data
		 */
		void save(CFile* saver);

		/** Real part of a complex128_t vector */
		SGVector<float64_t> get_real();

		/** Imag part of a complex128_t vector */
		SGVector<float64_t> get_imag();

		/** Create SGMatrix from linear vector
		 *
		 * @param vector source vector
		 * @param nrows number of rows
		 * @param ncols number of cols
		 * @param fortran_order order of stroing matrix in linear vector
		 *	true - column-major order (FORTRAN, MATLAB, R)
		 *	false - row-major order (C, Python)
		 * @return matrix
		 */
		static SGMatrix<T> convert_to_matrix(SGVector<T> vector, index_t nrows, index_t ncols, bool fortran_order);


		/** Create matrix from linear vector
		 *
		 * @param matrix destination memory
		 * @param nrows number of rows
		 * @param ncols number of cols
		 * @param vector source vector
		 * @param vlen lenght of source vector
		 * @param fortran_order order of stroing matrix in linear vector
		 *	true - column-major order (FORTRAN, MATLAB, R)
		 *	false - row-major order (C, Python)
		 * @return matrix
		 */
		static void convert_to_matrix(T*& matrix, index_t nrows, index_t ncols, const T* vector, int32_t vlen, bool fortran_order);
#endif // #ifndef SWIG // SWIG should skip this part
	protected:
		/** needs to be overridden to copy data */
		virtual void copy_data(const SGReferencedData &orig);

		/** needs to be overridden to initialize empty data */
		virtual void init_data();

		/** needs to be overridden to free data */
		virtual void free_data();

	public:
		/** vector  */
		T* vector;
		/** length of vector  */
		index_t vlen;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<> void SGVector<float64_t>::vec1_plus_scalar_times_vec2(float64_t* vec1,
				const float64_t scalar, const float64_t* vec2, int32_t n);

template<> void SGVector<float32_t>::vec1_plus_scalar_times_vec2(float32_t* vec1,
				const float32_t scalar, const float32_t* vec2, int32_t n);
#endif // DOXYGEN_SHOULD_SKIP_THIS
}
#endif // __SGVECTOR_H__
