/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <unsupported/Eigen/CXX11/Tensor>

namespace shogun::graph::op {
	template <typename T>
	void subtract_kernel_implementation_avx(
	    void* input1, void* input2, void* output, const size_t size);

	template <typename T>
	void subtract_kernel_implementation_avx(
	    void* input1, void* input2, void* output, const size_t size)
	{
		Eigen::TensorMap<Eigen::Tensor<T, 1>> A(static_cast<T*>(input1), size);
		Eigen::TensorMap<Eigen::Tensor<T, 1>> B(static_cast<T*>(input2), size);
		Eigen::TensorMap<Eigen::Tensor<T, 1>> Out(static_cast<T*>(output), size);

		Out.device(Eigen::DefaultDevice{}) = A - B;
	}

	template <>
	void subtract_kernel_implementation_avx<bool>(
	    void* input1, void* input2, void* output, const size_t size)
	{
		std::transform(
		    static_cast<const bool*>(input1),
		    static_cast<const bool*>(input1) + size,
		    static_cast<const bool*>(input2), static_cast<bool*>(output),
		    std::minus<bool>());
	}

	template void subtract_kernel_implementation_avx<int8_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_avx<int16_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_avx<int32_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_avx<int64_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_avx<uint8_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_avx<uint16_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_avx<uint32_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_avx<uint64_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_avx<float>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_avx<double>(void*, void*, void*, const size_t);
}