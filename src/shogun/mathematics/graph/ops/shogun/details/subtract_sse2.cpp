/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <Eigen/Core>

namespace shogun::graph::op {
	template <typename T>
	void subtract_kernel_implementation_sse2(
	    void* input1, void* input2, void* output, const size_t size);

	template <typename T>
	void subtract_kernel_implementation_sse2(
	    void* input1, void* input2, void* output, const size_t size)
	{
		Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> A(static_cast<T*>(input1), size);
		Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> B(static_cast<T*>(input2), size);
		Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> Out(static_cast<T*>(output), size);

		Out = A.array() - B.array();
	}

	template <>
	void subtract_kernel_implementation_sse2<bool>(
	    void* input1, void* input2, void* output, const size_t size)
	{
		std::transform(
		    static_cast<const bool*>(input1),
		    static_cast<const bool*>(input1) + size,
		    static_cast<const bool*>(input2), static_cast<bool*>(output),
		    std::minus<bool>());
	}

	template void subtract_kernel_implementation_sse2<int8_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_sse2<int16_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_sse2<int32_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_sse2<int64_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_sse2<uint8_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_sse2<uint16_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_sse2<uint32_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_sse2<uint64_t>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_sse2<float>(void*, void*, void*, const size_t);
	template void subtract_kernel_implementation_sse2<double>(void*, void*, void*, const size_t);
}