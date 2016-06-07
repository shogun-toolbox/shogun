#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalgrefactor/BaseVector.h>
#include <memory>

#ifndef GPU_VECTOR_H__
#define GPU_VECTOR_H__

namespace shogun
{

template <class T>
struct GPU_Vector : public BaseVector<T>
{
    friend class GPUBackend;

private:
    struct GPUArray;
    std::unique_ptr<GPUArray> gpuarray;

public:

    GPU_Vector();
    GPU_Vector(const SGVector<T> &vector);

    GPU_Vector(const GPU_Vector<T> &vector);

    ~GPU_Vector();

    void init();

    GPU_Vector<T>& operator=(const GPU_Vector<T> &other);

    const bool onGPU() { return true; }

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
