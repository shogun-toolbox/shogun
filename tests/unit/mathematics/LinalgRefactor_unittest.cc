/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2016 Pan Deng
 */

#include <shogun/lib/config.h>
#include <shogun/mathematics/linalgrefactor/SGLinalg.h>
#include <shogun/lib/SGVector.h>
#include <memory>
#include <gtest/gtest.h>

using namespace shogun;

TEST(SGLinalg, CPU_Vector_convert)
{
    const index_t size = 10;
    SGVector<int32_t> a(size);
    for (index_t i = 0; i < size; ++i) a[i] = i;

    CPUVector<int32_t> a_CPU(a);

    for (index_t i = 0; i < size; ++i)
        EXPECT_NEAR(a[i], (a_CPU.CPUptr)[i], 1E-15);
}

TEST(SGLinalg, CPU_Vector_copy)
{
    const index_t size = 10;
    SGVector<int32_t> a(size);
    for (index_t i = 0; i < size; ++i) a[i] = i;

    CPUVector<int32_t> a_CPU(a);
    CPUVector<int32_t> b_CPU(a_CPU);

    for (index_t i = 0; i < size; ++i)
        EXPECT_NEAR((a_CPU.CPUptr)[i], (b_CPU.CPUptr)[i], 1E-15);
}

TEST(SGLinalg, CPUBackend_dot)
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

TEST(SGLinalg, GPU_Vector_copy)
{
    const index_t size=10;
    SGVector<int32_t> a(size), b(size);
        for (index_t i = 0; i < size; ++i) a[i] = i;

    GPUVector<int32_t> a_GPU(a);
    GPUVector<int32_t> b_GPU;
    b_GPU = a_GPU;
}

TEST(SGLinalg, GPUBackend_dot)
{
    const index_t size=10;
    SGVector<int32_t> a(size), b(size);
    a.set_const(1);
    b.set_const(2);

    GPUVector<int32_t> a_GPU(a);
    GPUVector<int32_t> b_GPU(b);

    std::unique_ptr<GPUBackend> ViennaCLBackend;
    ViennaCLBackend = std::unique_ptr<GPUBackend>(new GPUBackend);

    sg_linalg->set_gpu_backend(std::move(ViennaCLBackend));
    auto result = sg_linalg->dot(&a_GPU, &b_GPU);

    EXPECT_NEAR(result, 20.0, 1E-15);
}

#endif
