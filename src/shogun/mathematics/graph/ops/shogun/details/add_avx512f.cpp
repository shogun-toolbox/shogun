/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <Eigen/Core>

namespace shogun::graph::op {
	template <typename T>
	void add_kernel_implementation_avx512f(
	    void* input1, void* input2, void* output, const size_t size);

	template <typename T>
	void add_kernel_implementation_avx512f(
	    void* input1, void* input2, void* output, const size_t size)
	{
		Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> A(static_cast<T*>(input1), size);
		Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> B(static_cast<T*>(input2), size);
		Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> Out(static_cast<T*>(output), size);

		Out = A + B;
	}

	template <>
	void add_kernel_implementation_avx512f<bool>(
	    void* input1, void* input2, void* output, const size_t size)
	{
		std::transform(
		    static_cast<const bool*>(input1),
		    static_cast<const bool*>(input1) + size,
		    static_cast<const bool*>(input2), static_cast<bool*>(output),
		    std::plus<bool>());
	}

	template void add_kernel_implementation_avx512f<int8_t>(void*, void*, void*, const size_t);
	template void add_kernel_implementation_avx512f<int16_t>(void*, void*, void*, const size_t);
	template void add_kernel_implementation_avx512f<int32_t>(void*, void*, void*, const size_t);
	template void add_kernel_implementation_avx512f<int64_t>(void*, void*, void*, const size_t);
	template void add_kernel_implementation_avx512f<uint8_t>(void*, void*, void*, const size_t);
	template void add_kernel_implementation_avx512f<uint16_t>(void*, void*, void*, const size_t);
	template void add_kernel_implementation_avx512f<uint32_t>(void*, void*, void*, const size_t);
	template void add_kernel_implementation_avx512f<uint64_t>(void*, void*, void*, const size_t);
	template void add_kernel_implementation_avx512f<float>(void*, void*, void*, const size_t);
	template void add_kernel_implementation_avx512f<double>(void*, void*, void*, const size_t);
}