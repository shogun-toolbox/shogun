#ifndef GPUMemory_ViennaCL_H__
#define GPUMemory_ViennaCL_H__

#ifdef HAVE_CXX11
#ifdef HAVE_VIENNACL

#include <viennacl/vector.hpp>

namespace shogun
{

template <typename T>
struct GPUMemoryViennaCL : public GPUMemoryBase<T>
{
	typedef viennacl::backend::mem_handle VCLMemoryArray;
	typedef viennacl::vector_base<T, std::size_t, std::ptrdiff_t> VCLVectorBase;

	GPUMemoryViennaCL()
	{
		m_data = NULL;
		m_len = 0;
		m_offset = 0;
	};

	GPUMemoryViennaCL(const SGVector<T>& vector):m_len(vector.vlen), m_offset(0)
	{
		viennacl::backend::memory_create(*m_data, sizeof(T)*m_len,
				viennacl::context());

		viennacl::backend::memory_write(*m_data, 0, m_len*sizeof(T), vector.vector);
	}

	void transfer_to_CPU(T* data) const
	{
		viennacl::backend::memory_read(*m_data, m_offset*sizeof(T), m_len*sizeof(T),
			data);
	}

	/** The data */
	inline VCLVectorBase data()
	{
		return vcl_vector();
	}

private:
	/** Returns a ViennaCL vector wrapped around the data of this vector. Can be
	 * used to call native ViennaCL methods on this vector
	 */
	VCLVectorBase vcl_vector()
	{
		return VCLVectorBase(*m_data, m_len, m_offset, 1);
	}

    VCLMemoryArray* m_data;
	index_t m_len;
	index_t m_offset;
};

}
#endif // HAVE_VIENNACL
#endif // HAVE_CXX11

#endif //GPUMemory_ViennaCL_H__
