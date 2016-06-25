#ifndef GPUMemory_Base_H__
#define GPUMemory_Base_H__

namespace shogun
{

template <typename T>
struct GPUMemoryBase
{
	GPUMemoryBase()
	{
	}

	virtual GPUMemoryBase<T>* clone_vector(const SGVector<T>& vector) const = 0;

	virtual void from_gpu(T* data) const = 0;
};

}

#endif //GPUMemory_Base_H__
