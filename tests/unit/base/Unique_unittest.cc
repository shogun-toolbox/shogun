#include <shogun/base/some.h>
#include <shogun/kernel/GaussianKernel.h>
#include <gtest/gtest.h>

#include "Unique_unittest.h"

#ifdef HAVE_CXX11
using namespace shogun;

class SomeTestingClassWithUnique::Self
{
public:
    int data;
};

void SomeTestingClassWithUnique::set(int value)
{
    self->data = value;
}

int SomeTestingClassWithUnique::get()
{
    return self->data;
}

TEST(Unique,basic)
{
    SomeTestingClassWithUnique object;
    int value = rand();
    object.set(value);
    EXPECT_EQ(value, object.get());
}
#endif
