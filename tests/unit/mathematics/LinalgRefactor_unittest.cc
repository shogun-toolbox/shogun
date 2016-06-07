#include <shogun/lib/config.h>
#include <shogun/mathematics/linalgrefactor/linalgRefactor.h>
#include <shogun/lib/SGVector.h>
#include <memory>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LinalgRefactor, CPU_Vector_convert)
{
    const index_t size = 10;
    SGVector<int32_t> a(size);
    for (index_t i = 0; i < size; ++i) a[i] = i;

    CPUVector<int32_t> a_CPU(a);

    for (index_t i = 0; i < size; ++i)
        EXPECT_NEAR(a[i], (a_CPU.CPUptr)[i], 1E-15);
}

TEST(LinalgRefactor, CPU_Vector_copy)
{
    const index_t size = 10;
    SGVector<int32_t> a(size);
    for (index_t i = 0; i < size; ++i) a[i] = i;

    CPUVector<int32_t> a_CPU(a);
    CPUVector<int32_t> b_CPU(a_CPU);

    for (index_t i = 0; i < size; ++i)
        EXPECT_NEAR((a_CPU.CPUptr)[i], (b_CPU.CPUptr)[i], 1E-15);
}

TEST(LinalgRefactor, CPUBackend_dot)
{
    const index_t size=10;
    SGVector<int32_t> a(size), b(size);
    a.set_const(1);
    b.set_const(2);

    CPUVector<int32_t> a_CPU(a);
    CPUVector<int32_t> b_CPU(b);

    auto result = sg_linalg->dot(&a_CPU, &b_CPU);

    EXPECT_NEAR(result, 20.0, 1E-15);
}

#ifdef HAVE_VIENNACL

TEST(LinalgRefactor, GPU_Vector_copy)
{
    const index_t size=10;
    SGVector<int32_t> a(size), b(size);
        for (index_t i = 0; i < size; ++i) a[i] = i;

    GPU_Vector<int32_t> a_GPU(a);
    GPU_Vector<int32_t> b_GPU;
    b_GPU = a_GPU;
}

TEST(LinalgRefactor, GPUBackend_dot)
{
    const index_t size=10;
    SGVector<int32_t> a(size), b(size);
    a.set_const(1);
    b.set_const(2);

    GPU_Vector<int32_t> a_GPU(a);
    GPU_Vector<int32_t> b_GPU(b);

    GPUBackend ViennaCLBackend;

    sg_linalg->set_gpu_backend(&ViennaCLBackend);
    auto result = sg_linalg->dot(&a_GPU, &b_GPU);

    EXPECT_NEAR(result, 20.0, 1E-15);
}

#endif
