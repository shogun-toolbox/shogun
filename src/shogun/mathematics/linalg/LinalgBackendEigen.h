#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/LinalgBackendBase.h>

#include <iostream>

#ifndef Linalg_Backend_Eigen_H__
#define Linalg_Backend_Eigen_H__

#ifdef HAVE_CXX11

namespace shogun
{

class LinalgBackendEigen : public LinalgBackendBase
{
public:
    // macro to avoid templating
    #define BACKEND_GENERIC_DOT(Type) \
	Type dot(const SGVector<Type>& a, const SGVector<Type>& b) const \
    {  \
		std::cerr << "cpu.dot - cpu.backend" << std::endl; \
		return dot_impl(a, b);  \
    }

    BACKEND_GENERIC_DOT(float32_t);
    BACKEND_GENERIC_DOT(float64_t);
    BACKEND_GENERIC_DOT(int32_t);

	/** @return object name */
	virtual const char* get_name() const { return "Eigen3"; }

private:
	template <typename T>
	T dot_impl(const SGVector<T>& a, const SGVector<T>& b) const
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
		Eigen::Map<VectorXt> vec_a(a.vector, a.vlen);
		Eigen::Map<VectorXt> vec_b(b.vector, b.vlen);
		return vec_a.dot(vec_b);
	}

};

}

#endif // HAVE_CXX11

#endif //Linalg_Backend_Eigen_H__
