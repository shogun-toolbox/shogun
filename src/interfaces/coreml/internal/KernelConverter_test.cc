#include <gtest/gtest.h>
#include "internal/KernelConverter.h"

#include <shogun/kernel/LinearKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/SigmoidKernel.h>
#include <shogun/kernel/normalizer/IdentityKernelNormalizer.h>

#include "format/SVM.pb.h"

using namespace shogun;
using namespace shogun::coreml;

TEST(LinearKernel, convert)
{
    auto lk = some<CLinearKernel>();
    auto descr = KernelConverter::convert(lk.get());
}

TEST(GaussianKernel, convert)
{
    auto k = some<CGaussianKernel>();
    auto descr = KernelConverter::convert(k.get());

    ASSERT_TRUE(descr->has_rbfkernel());

    auto rbf = descr->rbfkernel();
    ASSERT_EQ(k->get_width(), rbf.gamma());
}

TEST(PolyKernel, convert)
{
    auto k = std::make_shared<CPolyKernel>(10, 3, 2.0, 2.2);
    k->set_normalizer(new CIdentityKernelNormalizer());
    auto descr = KernelConverter::convert(k.get());

    ASSERT_TRUE(descr->has_polykernel());

    auto pk = descr->polykernel();
    ASSERT_EQ(k->get<float64_t>("gamma"), pk.gamma());
    ASSERT_EQ(k->get<float64_t>("c"), pk.c());
    ASSERT_EQ(k->get<int32_t>("degree"), pk.degree());
}

TEST(SigmoidKernel, convert)
{
    auto k = std::make_shared<CSigmoidKernel>(10, 55.5, 2.3);
    auto descr = KernelConverter::convert(k.get());

    ASSERT_TRUE(descr->has_sigmoidkernel());

    auto sk = descr->sigmoidkernel();
    ASSERT_EQ(k->get<float64_t>("gamma"), sk.gamma());
    ASSERT_EQ(k->get<float64_t>("coef0"), sk.c());

}
