#include <shogun/mathematics/linalgrefactor/GPUVector.h>
#include <shogun/mathematics/linalgrefactor/GPUArray.h>

namespace shogun
{

template <class T>
GPUVector<T>::GPUVector()
{
	init();
}

template <class T>
GPUVector<T>::GPUVector(const SGVector<T> &vector)
{
	init();
	vlen = vector.vlen;

#ifdef HAVE_VIENNACL
	gpuarray = std::unique_ptr<GPUArray>(new GPUArray(vector));
#else
	SG_SERROR("User did not register GPU backend. \n");
#endif
}

template <class T>
GPUVector<T>::GPUVector(const GPUVector<T> &vector)
{
	init();
	vlen = vector.vlen;
	offset = vector.offset;
#ifdef HAVE_VIENNACL
	gpuarray = std::unique_ptr<GPUArray>(new GPUArray(*(vector.gpuarray)));
#else
	SG_SERROR("User did not register GPU backend. \n");
#endif
}

template <class T>
void GPUVector<T>::init()
{
	vlen = 0;
	offset = 0;
}

template <class T>
GPUVector<T>& GPUVector<T>::operator=(const GPUVector<T> &other)
{
	// check for self-assignment
	if(&other == this)
	{
		return *this;
	}

	// reuse storage when possible
	gpuarray.reset(new GPUArray(*(other.gpuarray)));
	vlen = other.vlen;
	return *this;
}

template <class T>
GPUVector<T>::~GPUVector()
{
}

template struct GPUVector<int32_t>;
template struct GPUVector<float32_t>;

}
