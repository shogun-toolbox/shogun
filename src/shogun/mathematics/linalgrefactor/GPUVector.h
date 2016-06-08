#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalgrefactor/BaseVector.h>
#include <memory>

#ifndef GPU_Vector_H__
#define GPU_Vector_H__

namespace shogun
{

template <class T>
struct GPUVector : public BaseVector<T>
{
	friend class GPUBackend;

private:
	struct GPUArray;
	std::unique_ptr<GPUArray> gpuarray;

public:

	GPUVector();
	GPUVector(const SGVector<T> &vector);

	GPUVector(const GPUVector<T> &vector);

	~GPUVector();

	void init();

	GPUVector<T>& operator=(const GPUVector<T> &other);

	inline bool onGPU()
	{
		return true;
	}

public:
	/** Vector length */
	index_t vlen;

	/** Offset for the memory segment, i.e the data of the vector
	 * starts at vector+offset
	 */
	index_t offset;
};

}

#endif
