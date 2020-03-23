/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <unsupported/Eigen/CXX11/Tensor>

namespace shogun::graph::op {
	template <typename T>
	void subtract_kernel_implementation_sse41(
	    void* input1, void* input2, void* output, const size_t size);

	template <typename T>
	void subtract_kernel_implementation_sse41(
	    void* input1, void* input2, void* output, const size_t size)
	{
		Eigen::TensorMap<Eigen::Tensor<T, 1>> A(static_cast<T*>(input1), size);
		Eigen::TensorMap<Eigen::Tensor<T, 1>> B(static_cast<T*>(input2), size);
		Eigen::TensorMap<Eigen::Tensor<T, 1>> Out(static_cast<T*>(output), size);

		Out.device(Eigen::DefaultDevice{}) = A - B;
	}

	template void subtract_kernel_implementation_sse41<int32_t>(void*, void*, void*, const size_t);
}