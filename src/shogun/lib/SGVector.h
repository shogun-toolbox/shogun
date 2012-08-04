/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */
#ifndef __SGVECTOR_H__
#define __SGVECTOR_H__

#include <shogun/io/SGIO.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGReferencedData.h>

namespace shogun
{
/** @brief shogun vector */
template<class T> class SGVector : public SGReferencedData
{
	public:
		/** default constructor */
		SGVector();

		/** constructor for setting params */
		SGVector(T* v, index_t len, bool ref_counting=true);

		/** constructor to create new vector in memory */
		SGVector(index_t len, bool ref_counting=true);

		/** copy constructor */
		SGVector(const SGVector &orig);

		/** empty destructor */
		virtual ~SGVector();

		/** cast to pointer */
		operator T*() { return vector; };

		/** fill vector with zeros */
		void zero();

		/** set vector to a constant
		 *
		 * @param const_elem - value to set vector to
		 */
		void set_const(T const_elem);

		/** range fill a vector with start...start+len-1
		 * 
		 * @param start - value to be assigned to first element of vector
		 */
		void range_fill(T start=0);

		/** create random vector
		 *
		 * @param min_value [min_value,max_value]
		 * @param max_value
		 */
		void random(T min_value, T max_value);

		/** random permutate */
		void randperm();

		/** clone vector */
		SGVector<T> clone() const;

		/** clone vector */
		static T* clone_vector(const T* vec, int32_t len)
		{
			T* result = SG_MALLOC(T, len);
			memcpy(result, vec, sizeof(T)*len);
			return result;
		}

		/** fill vector */
		static void fill_vector(T* vec, int32_t len, T value)
		{
			for (int32_t i=0; i<len; i++)
				vec[i]=value;
		}

		/** range fill vector */
		static void range_fill_vector(T* vec, int32_t len, T start=0)
		{
			for (int32_t i=0; i<len; i++)
				vec[i]=i+start;
		}

		/** random vector */
		static void random_vector(T* vec, int32_t len, T min_value, T max_value);

		/** random permatutaion */
		static void randperm(T* perm, int32_t n);

		/** permute */
		static void permute(T* vec, int32_t n);

		/** get vector element at index
		 *
		 * @param index index
		 * @return vector element at index
		 */
		const T& get_element(index_t index);

		/** set vector element at index 'index' return false in case of trouble
		 *
		 * @param p_element vector element to set
		 * @param index index
		 * @return if setting was successful
		 */
		void set_element(const T& p_element, index_t index);

		/** resize vector
		 *
		 * @param n new size
		 * @return if resizing was successful
		 */
		void resize_vector(int32_t n);

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

		/** add vector to current vector
		 *
		 * @param x add vector x to current vector
		 */
		void add(const SGVector<T> x);

		/** add scalar to current vector
		 *
		 * @param x add vector x to current vector
		 */
		void add(const T x);

		SGVector<T> operator+ (SGVector<T> x)
		{
			ASSERT(x.vector && vector);
			ASSERT(x.vlen == vlen);

			SGVector<T> result=clone();
			result.add(x);
			return result;
		}

		SGVector<T> operator+= (SGVector<T> x)
		{
			add(x);
			return *this;
		}

		static void permute_vector(SGVector<T> vec);

		/** create a random permutation in place */
		void permute();


		/** resize array from old_size to new_size (keeping as much array
		 * content as possible intact)
		 */
		static inline void resize(T* &data, int64_t old_size, int64_t new_size)
		{
			if (old_size==new_size)
				return;

			data = SG_REALLOC(T, data, new_size);
		}

		/// || x ||_2
		static T twonorm(const T* x, int32_t len);

		/// || x ||_1
		static float64_t onenorm(T* x, int32_t len);

		/// || x ||_q^q
		static T qsq(T* x, int32_t len, float64_t q);

		/// || x ||_q
		static T qnorm(T* x, int32_t len, float64_t q);

//		/// x=x+alpha*y
//		static inline void vec1_plus_scalar_times_vec2(T* vec1,
//				T scalar, const T* vec2, int32_t n)
//		{
//			for (int32_t i=0; i<n; i++)
//				vec1[i]+=scalar*vec2[i];
//		}

		/// x=x+alpha*y (blas optimized)
		static void vec1_plus_scalar_times_vec2(float64_t* vec1,
				const float64_t scalar, const float64_t* vec2, int32_t n);

		/// x=x+alpha*y (blas optimized)
		static void vec1_plus_scalar_times_vec2(float32_t* vec1,
				const float32_t scalar, const float32_t* vec2, int32_t n);

		/// compute dot product between v1 and v2 (blas optimized)
		static inline float64_t dot(const bool* v1, const bool* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((v1[i]) ? 1 : 0) * ((v2[i]) ? 1 : 0);
			return r;
		}

		/// compute dot product between v1 and v2 (blas optimized)
		static inline floatmax_t dot(const floatmax_t* v1, const floatmax_t* v2, int32_t n)
		{
			floatmax_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=v1[i]*v2[i];
			return r;
		}


		/// compute dot product between v1 and v2 (blas optimized)
		static float64_t dot(const float64_t* v1, const float64_t* v2, int32_t n);

		/// compute dot product between v1 and v2 (blas optimized)
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
		/// compute dot product between v1 and v2 (for 64bit ints)
		static inline float64_t dot(
			const int64_t* v1, const int64_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 32bit ints)
		static inline float64_t dot(
			const int32_t* v1, const int32_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 32bit unsigned ints)
		static inline float64_t dot(
			const uint32_t* v1, const uint32_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 16bit unsigned ints)
		static inline float64_t dot(
			const uint16_t* v1, const uint16_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 16bit unsigned ints)
		static inline float64_t dot(
			const int16_t* v1, const int16_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 8bit (un)signed ints)
		static inline float64_t dot(
			const char* v1, const char* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 8bit (un)signed ints)
		static inline float64_t dot(
			const uint8_t* v1, const uint8_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 8bit (un)signed ints)
		static inline float64_t dot(
			const int8_t* v1, const int8_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2
		static inline float64_t dot(
			const float64_t* v1, const char* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute vector multiplication
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

		/// add scalar to vector inplace
		static inline void add_scalar(T alpha, T* vec, int32_t len)
		{
			for (int32_t i=0; i<len; i++)
				vec[i]+=alpha;
		}

		/// scale vector inplace
		static inline void scale_vector(T alpha, T* vec, int32_t len)
		{
			for (int32_t i=0; i<len; i++)
				vec[i]*=alpha;
		}

		/// return sum(vec)
		static inline T sum(T* vec, int32_t len)
		{
			T result=0;
			for (int32_t i=0; i<len; i++)
				result+=vec[i];

			return result;
		}

		/// return sum(vec)
		static inline T sum(SGVector<T> vec)
		{
			return sum(vec.vector, vec.vlen);
		}


		/** @return min(vec) */
		static T min(T* vec, int32_t len);

		/** @return max(vec) */
		static T max(T* vec, int32_t len);

		/// return arg_max(vec)
		static inline int32_t arg_max(T * vec, int32_t inc, int32_t len, T * maxv_ptr = NULL)
		{
			ASSERT(len > 0 || inc > 0);

			T maxv = vec[0];
			int32_t maxIdx = 0;

			for (int32_t i = 1, j = inc ; i < len ; i++, j += inc)
			{
				if (vec[j] > maxv)
					maxv = vec[j], maxIdx = i;
			}

			if (maxv_ptr != NULL)
				*maxv_ptr = maxv;

			return maxIdx;
		}

		/// return arg_min(vec)
		static inline int32_t arg_min(T * vec, int32_t inc, int32_t len, T * minv_ptr = NULL)
		{
			ASSERT(len > 0 || inc > 0);

			T minv = vec[0];
			int32_t minIdx = 0;

			for (int32_t i = 1, j = inc ; i < len ; i++, j += inc)
			{
				if (vec[j] < minv)
					minv = vec[j], minIdx = i;
			}

			if (minv_ptr != NULL)
				*minv_ptr = minv;

			return minIdx;
		}

		/// return sum(abs(vec))
		static T sum_abs(T* vec, int32_t len);

		/// return sum(abs(vec))
		static bool fequal(T x, T y, float64_t precision=1e-6);

		/* performs a inplace unique of a vector of type T using quicksort
		 * returns the new number of elements */
		static int32_t unique(T* output, int32_t size);

		/** display array size */
		void display_size() const;

		/** display vector */
		void display_vector(const char* name="vector") const;

		/// display vector (useful for debugging)
		static void display_vector(
			const T* vector, int32_t n, const char* name="vector",
			const char* prefix="");

		static void display_vector(
			const SGVector<T>, const char* name="vector",
			const char* prefix="");


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
}
#endif // __SGVECTOR_H__
