#include <shogun/lib/config.h>

#include <shogun/mathematics/linalg/linalgRefactor.h>
#include <shogun/lib/SGVector.h>
#include <algorithm>
#include <memory>
#include <gtest/gtest.h>

#include <shogun/mathematics/eigen3.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUVector.h>
#endif // HAVE_VIENNACL

using namespace shogun;

TEST(LinalgRefactor, CPU_Vector_convert)
{
    const index_t size = 10;
    SGVector<int32_t> a(size);
    for (index_t i = 0; i < size; ++i) a[i] = i;

    CPU_Vector<int32_t> a_CPU(a);

    for (index_t i = 0; i < size; ++i)
        EXPECT_NEAR(a[i], a_CPU.vec[i], 1E-15);
}


TEST(LinalgRefactor, CPUBackend_dot)
{
    const index_t size=10;
    SGVector<int32_t> a(size), b(size);
    a.set_const(1);
    b.set_const(2);

    CPU_Vector<int32_t> a_CPU(a);
    CPU_Vector<int32_t> b_CPU(b);
    BaseVector<int32_t>* a_base = static_cast<BaseVector<int32_t>*>(&a_CPU);
    BaseVector<int32_t>* b_base = static_cast<BaseVector<int32_t>*>(&b_CPU);

    std::shared_ptr<BaseVector<int32_t>> a_ptr = std::make_shared<BaseVector<int32_t> >(*a_base);
    std::shared_ptr<BaseVector<int32_t>> b_ptr = std::make_shared<BaseVector<int32_t> >(*b_base);
    CPUBackend eigenBackend;
    LinalgRefactor lr(std::make_shared<CPUBackend>(eigenBackend));

    int32_t result = eigenBackend.dot(*static_cast<CPU_Vector<int32_t>*>(a_ptr.get()),
                                *static_cast<CPU_Vector<int32_t>*>(b_ptr.get()));
    //int32_t result = lr.dot(std::make_shared<BaseVector<int32_t> >(*a_base),
    //                        std::make_shared<BaseVector<int32_t> >(*b_base));

    EXPECT_NEAR(result, 20.0, 1E-15);
}
