#ifndef GPUMemory_ViennaCL_H__
#define GPUMemory_ViennaCL_H__

#ifdef HAVE_CXX11
#ifdef HAVE_VIENNACL

#include <viennacl/vector.hpp>
#include <memory>

namespace shogun
{

template <typename T>
struct GPUMemoryViennaCL : public GPUMemoryBase<T>
{
	typedef viennacl::backend::mem_handle VCLMemoryArray;
	typedef viennacl::vector_base<T, std::size_t, std::ptrdiff_t> VCLVectorBase;

	GPUMemoryViennaCL()
	{
		init();
	};

	GPUMemoryViennaCL(GPUMemoryBase<T>* vector)
	{
		GPUMemoryViennaCL<T>* temp_vec = static_cast<GPUMemoryViennaCL<T>*>(vector);
		init();
		m_data = temp_vec->m_data;
		m_len = temp_vec->m_len;
		m_offset = temp_vec->m_offset;
	};

	GPUMemoryViennaCL(const SGVector<T>& vector)
	: m_data(new VCLMemoryArray())
	{
		init();
		m_len = vector.vlen;

		viennacl::backend::memory_create(*m_data, sizeof(T)*m_len,
				viennacl::context());

		viennacl::backend::memory_write(*m_data, 0, m_len*sizeof(T), vector.vector);
	}

	GPUMemoryBase<T>* clone_vector(const SGVector<T>& vector) const
	{
		std::shared_ptr<GPUMemoryViennaCL<T>> temp_vector;
		temp_vector = std::shared_ptr<GPUMemoryViennaCL<T>>(new GPUMemoryViennaCL<T>(vector));
		return temp_vector.get();
	}

	void from_gpu(T* data) const
	{
		viennacl::backend::memory_read(*m_data, m_offset*sizeof(T), m_len*sizeof(T),
			data);
	}

	/** The data */
	VCLVectorBase data()
	{
		return vcl_vector();
	}

private:
	void init()
	{
		m_len = 0;
		m_offset = 0;
	}

	/** Returns a ViennaCL vector wrapped around the data of this vector. Can be
	 * used to call native ViennaCL methods on this vector
	 */
	VCLVectorBase vcl_vector()
	{
		return VCLVectorBase(*m_data, m_len, m_offset, 1);
	}

	std::shared_ptr<VCLMemoryArray> m_data;
	index_t m_len;
	index_t m_offset;
};

}
#endif // HAVE_VIENNACL
#endif // HAVE_CXX11

#endif //GPUMemory_ViennaCL_H__
