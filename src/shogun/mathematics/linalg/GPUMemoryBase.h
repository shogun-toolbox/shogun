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

	virtual void transfer_to_CPU(T* data) const = 0;
};

}

#endif //GPUMemory_Base_H__
