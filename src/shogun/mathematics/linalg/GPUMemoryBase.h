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

	virtual void from_gpu(T* data) const = 0;
};

}

#endif //GPUMemory_Base_H__
