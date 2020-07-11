/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Fernando Iglesias, Viktor Gal,
 *          Sergey Lisitsyn, Sanuj Sharma, Soumyajit De, Koen van de Sande,
 *          Pan Deng, Chiyuan Zhang, Thoralf Klein, Khaled Nasr,
 *          Evgeniy Andreev, Michele Mazzoni, Bjoern Esser, Yuyu Zhang
 */
#ifndef __SGVECTOR_H__
#define __SGVECTOR_H__

#include <shogun/lib/config.h>

#include <shogun/base/macros.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGReferencedData.h>
#include <shogun/util/iterators.h>
#include <shogun/mathematics/linalg/GPUMemoryBase.h>

#include <algorithm>
#include <atomic>
#include <initializer_list>
#include <memory>
#include <string>

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
	class File;

/** @brief shogun vector */
template<class T> class SGVector : public SGReferencedData
{
	friend class LinalgBackendEigen;

	public:
		typedef RandomIterator<T> iterator;
		typedef ConstRandomIterator<T> const_iterator;

	public:
		typedef Eigen::Matrix<T,-1,1,0,-1,1> EigenVectorXt;
		typedef Eigen::Matrix<T,1,-1,0x1,1,-1> EigenRowVectorXt;

		typedef Eigen::Map<EigenVectorXt,0,Eigen::Stride<0,0> > EigenVectorXtMap;
		typedef Eigen::Map<EigenRowVectorXt,0,Eigen::Stride<0,0> > EigenRowVectorXtMap;

		/** The scalar type of the vector */
		typedef T Scalar;

		/** Default constructor */
		SGVector();

		/** Constructor for setting params */
		SGVector(T* v, index_t len, bool ref_counting=true);

		/** Wraps a vector around an existing memory segment with an offset */
		SGVector(T* m, index_t len, index_t offset);

		/** Constructor to create new vector in memory */
		SGVector(index_t len, bool ref_counting=true);

		/** Constructor to create new vector from a SGMatrix */
		SGVector(SGMatrix<T> matrix);

		/** Construct SGVector from GPU memory.
		 *
		 * @param vector GPUMemoryBase pointer
		 * @param len length of the data in vector
		 * @see GPUMemoryBase
		 */
		SGVector(GPUMemoryBase<T>* vector, index_t len);

		/** Copy constructor */
		SGVector(const SGVector &orig);

		/** Move constructor */
		SGVector(SGVector&& orig) noexcept;

		/** Check whether data is stored on GPU
		 *
		 * @return true if vector is on GPU
		 */
#ifndef SWIG
		SG_FORCED_INLINE
#endif
		bool on_gpu() const
		{
#ifdef HAVE_VIENNACL
			return gpu_ptr != NULL;
#else
			return false;
#endif
		}

#ifndef SWIG // SWIG should skip this part

		/** The container type for a given template argument */
		template <typename ST> using container_type = SGVector<ST>;

		/** Construct SGVector from InputIterator list */
		template<typename InputIt>
		SGVector(InputIt beginIt, InputIt endIt):
			SGReferencedData(true),
			vlen(std::distance(beginIt, endIt)),
			gpu_ptr(nullptr)
		{
			vector = SG_ALIGNED_MALLOC(T, vlen, alignment::container_alignment);
			std::copy(beginIt, endIt, vector);
#ifdef HAVE_VIENNACL
            m_on_gpu.store(false, std::memory_order_release);
#endif
		}

		/** Construct SGVector from initializer list */
		SGVector(std::initializer_list<T> il);

		/** Wraps a matrix around the data of an Eigen3 column vector */
		SGVector(EigenVectorXt& vec);

		/** Wraps a matrix around the data of an Eigen3 row vector */
		SGVector(EigenRowVectorXt& vec);

		/** Wraps an Eigen3 column vector around the data of this matrix */
		operator EigenVectorXtMap() const;

		/** Wraps an Eigen3 row vector around the data of this matrix */
		operator EigenRowVectorXtMap() const;

		/** @return a (copied) typed vector with same content */
		template <class X>
		SGVector<X> as() const
		{
			SGVector<X> v(vlen);
			std::copy_n(vector, vlen, v.begin());
			return v;
		}
#endif // SWIG

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
		~SGVector() override;

		/** Size */
		inline int32_t size() const { return vlen; }

		/** Data pointer */
		inline T* data() const
		{
			assert_on_cpu();
			return vector;
		}

		/** Returns an iterator to the first element of the container. */
		iterator begin() noexcept { return iterator(vector); }

		/** Returns an iterator to the element following the last element of the container. */
		iterator end() noexcept { return iterator(vector + vlen); }

		/** Returns a const iterator to the first element of the container. */
		const_iterator begin() const noexcept { return const_iterator(vector); }

		/** Returns a const iterator to the element following the last element of the container. */
		const_iterator end() const noexcept { return const_iterator(vector + vlen); }

		SGVector<T>& operator=(const SGVector<T>&);

		/** Check for pointer identity */
		SG_FORCED_INLINE bool operator==(const SGVector<T>& other) const
		{
			if (vlen != other.vlen)
				return false;

			if (on_gpu())
			{
				if (!other.on_gpu())
					return false;
				if (gpu_ptr != other.gpu_ptr)
					return false;
			}

			if (vector != other.vector)
				return false;

			return true;
		}

		/** Cast to pointer */
		operator T*() { return vector; }

		/** Fill vector with zeros */
		void zero();

		/** Range fill a vector with start...start+len-1
		 *
		 * @param start - value to be assigned to first element of vector
		 */
		void range_fill(T start=0);

		/** For a sorted (ascending) vector, gets the index after the first
		 * element that is smaller than the given one
		 *
		 * @param element element to find index for
		 * @return index of the first element greater than given one
		 */
		index_t find_position_to_insert(T element);

		/** Clone vector */
		SGVector<T> clone() const;

		/** Clone vector */
		static T* clone_vector(const T* vec, int32_t len);

		/** Constructs a new empty vector with same dimensions as another 
		 * vector instance 
		 */
		static SGVector<T> empty_like(const SGVector<T>&);

		/** Fill vector */
		static void fill_vector(T* vec, int32_t len, T value);

		/** Range fill vector */
		static void range_fill_vector(T* vec, int32_t len, T start=0);

#endif // SWIG // SWIG should skip this part

		/** Get element at index
		 *
		 * @param index index
		 * @return element at index
		 */
		const T& get_element(index_t index)
		{
			return (*this)[index];
		}

		/** Set element at index
		 *
		 * @param el element to set
		 * @param index index
		 */
		void set_element(const T& el, index_t index)
		{
			(*this)[index]=el;
		}

#ifndef SWIG // SWIG should skip this part
		/** Resize vector, with zero padding
		 *
		 * @param n new size
		 */
		void resize_vector(int32_t n);

		/** Returns a view of the vector from l inclusive to h exclusive
		 *
		 * @param l slice start index (inclusive)
		 * @param h slice end index (exclusive)
		 */
		SGVector<T> slice(index_t l, index_t h) const;

		/** Operator overload for vector read only access
		 *
		 * @param index dimension to access
		 *
		 */
		inline const T& operator[](uint64_t index) const
		{
			assert_on_cpu();
			return vector[index];
		}

		/** Operator overload for vector read only access
		 *
		 * @param index dimension to access
		 *
		 */
		inline const T& operator[](int64_t index) const
		{
			assert_on_cpu();
			return vector[index];
		}

		/** Operator overload for vector read only access
		 *
		 * @param index dimension to access
		 *
		 */
		inline const T& operator[](uint32_t index) const
		{
			assert_on_cpu();
			return vector[index];
		}

		/** Operator overload for vector read only access
		 *
		 * @param index dimension to access
		 *
		 */
		inline const T& operator[](int32_t index) const
		{
			assert_on_cpu();
			return vector[index];
		}

		/** Operator overload for vector r/w access
		 *
		 * @param index dimension to access
		 *
		 */
		inline T& operator[](uint64_t index)
		{
			assert_on_cpu();
			return vector[index];
		}

		/** Operator overload for vector r/w access
		 *
		 * @param index dimension to access
		 *
		 */
		inline T& operator[](int64_t index)
		{
			assert_on_cpu();
			return vector[index];
		}

		/** Operator overload for vector r/w access
		 *
		 * @param index dimension to access
		 *
		 */
		inline T& operator[](uint32_t index)
		{
			assert_on_cpu();
			return vector[index];
		}

		/** Operator overload for vector r/w access
		 *
		 * @param index dimension to access
		 *
		 */
		inline T& operator[](int32_t index)
		{
			assert_on_cpu();
			return vector[index];
		}

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

		/** Equals method up to precision for vectors (element-wise)
		 * @param other vector to compare with
		 * @return false if any element differs or if sizes are different,
		 * true otherwise
		 */
		bool equals(const SGVector<T>& other) const;

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

		/** Performs a inplace unique of a vector of type T using quicksort
		 * returns the new number of elements
		 */
		static int32_t unique(T* output, int32_t size);

		/** Returns new vector with sorted unique elements of current */
		SGVector<T> unique();

		// get the values of the vector in a string representation
		std::string to_string() const;

		// dont use this this is just until display_vector is dropped
		static std::string to_string(const T* vector, index_t n);

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
		void load(const std::shared_ptr<File>& loader);

		/** Save vector to file
		 *
		 * @param saver File object via which to save data
		 */
		void save(const std::shared_ptr<File>& saver);

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
		void copy_data(const SGReferencedData &orig) override;

		/** needs to be overridden to initialize empty data */
		void init_data() override;

		/** needs to be overridden to free data */
		void free_data() override;

	private:
#ifdef HAVE_VIENNACL
		/** Atomic variable of vector on_gpu status */
		std::atomic<bool> m_on_gpu;
#endif

		/** Assert whether the data is on GPU
		 * and raise error if the data is on GPU
		 */
#ifndef SWIG
		SG_FORCED_INLINE
#endif
		void assert_on_cpu() const
		{
#ifdef HAVE_VIENNACL
            if (on_gpu())
				error("Direct memory access not possible when data is in GPU memory.");
#endif
		}

	public:
		/** Pointer to memory where vector data is stored */
		T* vector;
		/** Length of vector  */
		index_t vlen;
		/** GPU Vector structure. Stores pointer to the data on GPU. */
		std::shared_ptr<GPUMemoryBase<T>> gpu_ptr;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<> void SGVector<float64_t>::vec1_plus_scalar_times_vec2(float64_t* vec1,
				const float64_t scalar, const float64_t* vec2, int32_t n);

template<> void SGVector<float32_t>::vec1_plus_scalar_times_vec2(float32_t* vec1,
				const float32_t scalar, const float32_t* vec2, int32_t n);
#endif // DOXYGEN_SHOULD_SKIP_THIS
}
#endif // __SGVECTOR_H__
