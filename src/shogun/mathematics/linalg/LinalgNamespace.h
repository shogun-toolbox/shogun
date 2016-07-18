/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: 2016 Pan Deng, Soumyajit De, Heiko Strathmann, Viktor Gal
 */

#include <shogun/mathematics/linalg/LinalgBackendBase.h>
#include <shogun/mathematics/linalg/SGLinalg.h>

namespace shogun
{

namespace linalg
{

/** Infer the appropriate backend for linalg operations
 * from the input SGVector or SGMatrix(Container).
 *
 * @param a SGVector or SGMatrix
 * @return LinalgBackendBase pointer
 */
template <typename T, template <typename> class Container>
LinalgBackendBase* infer_backend(const Container<T>& a)
{
	if (a.on_gpu())
	{
		if (sg_linalg->get_gpu_backend())
			return sg_linalg->get_gpu_backend();
		else
		{
			SG_SERROR("Vector memory on GPU but no GPU backend registered. \
						This can happen if the GPU backend was de-activated \
						after memory has been transferred to GPU.\n");
			return NULL;
		}
	}
	else
		return sg_linalg->get_cpu_backend();
}

/** Infer the appropriate backend for linalg operations
 * from the input SGVector or SGMatrix(Container).
 * Raise error if the backends from the two Containers don't match.
 *
 * @param a the first SGVector/SGMatrix
 * @param b the second SGVector/SGMatrix
 * @return LinalgBackendBase pointer
 */
template <typename T, template <typename> class Container>
LinalgBackendBase* infer_backend(const Container<T>& a, const Container<T>& b)
{
	if (a.on_gpu() && b.on_gpu())
	{
		if (sg_linalg->get_gpu_backend())
			return sg_linalg->get_gpu_backend();
		else
		{
			SG_SERROR("Vector memory on GPU but no GPU backend registered. \
					  This can happen if the GPU backend was de-activated \
					  after memory has been transferred to GPU.\n");
			return NULL;
		}
	}
	else if (a.on_gpu() || b.on_gpu())
	{
		SG_SERROR("Cannot operate with first vector gpu (%d) and second vector gpu (%d).\n",
					a.on_gpu(), b.on_gpu());
		return NULL;
	}
	else
		return sg_linalg->get_cpu_backend();
}

/**
 * Performs the operation C = alpha*A + beta*B.
 * @param A first vector
 * @param B second vector
 * @param alpha constant to be multiplied by the first vector
 * @param beta constant to be multiplied by the second vector
 * @return The result vector
 */
template <typename T, template <typename> class Container>
Container<T> add(const Container<T>& a, const Container<T>& b, T alpha=1, T beta=1)
{
	REQUIRE(a.vlen == b.vlen, "Vectors should have same length! "
			"a(%d) vs b(%d)\n", a.vlen, b.vlen);

	return infer_backend(a, b)->add(a, b, alpha, beta);
}

/**
 * Vector dot-product that works with generic vectors.
 *
 * @param a first vector
 * @param b second vector
 * @return the dot product of \f$\mathbf{a}\f$ and \f$\mathbf{b}\f$, represented
 * as \f$\sum_i a_i b_i\f$
 */
template <typename T>
T dot(const SGVector<T>& a, const SGVector<T>& b)
{
	return infer_backend(a, b)->dot(a, b);
}

/**
 * Method that computes the sum of vectors of matrices
 *
 * @param vec a vector whose sum has to be computed
 * @return the vector sum \f$\sum_i a_i\f$
 */
template <typename T, template <typename> class Container>
T sum(const Container<T>& a)
{
	return infer_backend(a)->sum(a);
}

/**
 * Transfers data to GPU memory. Does nothing if no GPU backend registered.
 *
 * @param vector SGVector to be transferred
 * @return SGVector with vector on GPU if GPU backend is available
 * and a shallow-copy of SGVector with vector on CPU if GPU backend not available
 */
template <typename T>
SGVector<T> to_gpu(const SGVector<T>& vector)
{
	REQUIRE(!vector.on_gpu(), "The vector is already on GPU.\n");
	LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();
	if (gpu_backend)
		return SGVector<T>(gpu_backend->to_gpu(vector), vector.vlen);
	else
	{
		SG_SWARNING("Trying to access GPU memory without GPU backend registered.\n");
		return vector;
	}
}

/**
 * Transfers data to GPU memory. Does nothing if no GPU backend registered.
 *
 * @param vector SGMatrix to be transferred
 * @return SGMatrix with matrix on GPU if GPU backend is available
 * and a shallow-copy of SGMatrix with matrix on CPU if GPU backend not available
 */
template <typename T>
SGMatrix<T> to_gpu(const SGMatrix<T>& mat)
{
	REQUIRE(!mat.on_gpu(), "The matrix is already on GPU.\n");
	LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();
	if (gpu_backend)
		return SGMatrix<T>(gpu_backend->to_gpu(mat), mat.num_rows, mat.num_cols);
	else
	{
		SG_SWARNING("Trying to access GPU memory without GPU backend registered.\n");
		return mat;
	}
}

/**
 * Fetches data from GPU memory.
 *
 * @param vector SGVector to be transferred
 * @return SGVector with vector on CPU if GPU backend is still available
 * and a shallow-copy of SGVector with vector on GPU if GPU backend not available
 */
template <typename T>
SGVector<T> from_gpu(const SGVector<T>& vec)
{
	if (vec.on_gpu())
	{
		LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();
		if (gpu_backend)
		{
			typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type aligned_t;
			T* data;
			data = reinterpret_cast<T*>(SG_MALLOC(aligned_t, vec.size()));
			gpu_backend->from_gpu(vec, data);
			return SGVector<T>(data, vec.size());
		}
		else
		{
			SG_SERROR("Data memory on GPU but no GPU backend registered. \
						This can happen if the GPU backend was de-activated \
						after memory has been transferred to GPU.\n");
			return NULL;
		}
	}
	else
	{
		SG_SWARNING("The data is already on CPU.\n");
		return vec;
	}

}

/**
 * Fetches data from GPU memory.
 *
 * @param vector SGMatrix to be transferred
 * @return SGMatrix with matrix on CPU if GPU backend is still available
 * and a shallow-copy of SGMatrix with matrix on GPU if GPU backend not available
 */
template <typename T>
SGMatrix<T> from_gpu(const SGMatrix<T>& mat)
{
	if (mat.on_gpu())
	{
		LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();
		if (gpu_backend)
		{
			typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type aligned_t;
			T* data;
			data = reinterpret_cast<T*>(SG_MALLOC(aligned_t, mat.num_rows*mat.num_cols));
			gpu_backend->from_gpu(mat, data);
			return SGMatrix<T>(data, mat.num_rows, mat.num_cols);
		}
		else
		{
			SG_SERROR("Data memory on GPU but no GPU backend registered. \
						This can happen if the GPU backend was de-activated \
						after memory has been transferred to GPU.\n");
			return NULL;
		}
	}
	else
	{
		SG_SWARNING("The data is already on CPU.\n");
		return mat;
	}
}

}

}
