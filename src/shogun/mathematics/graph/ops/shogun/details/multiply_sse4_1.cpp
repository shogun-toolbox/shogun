/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <Eigen/Core>

namespace shogun::graph::op {
	template <typename T>
	void multiply_kernel_implementation_sse41(
	    void* input1, void* input2, void* output, const size_t size);

	template <typename T>
	void multiply_kernel_implementation_sse41(
	    void* input1, void* input2, void* output, const size_t size)
	{
		Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> A(static_cast<T*>(input1), size);
		Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> B(static_cast<T*>(input2), size);
		Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> Out(static_cast<T*>(output), size);

		Out = A.array() * B.array();
	}

	template void multiply_kernel_implementation_sse41<int32_t>(void*, void*, void*, const size_t);
}