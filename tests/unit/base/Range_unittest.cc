#include <shogun/base/range.h>
#include <shogun/lib/config.h>
#include <gtest/gtest.h>

#ifdef HAVE_CXX11
using namespace shogun;

TEST(Range, basic_upper)
{
    int other_i = 0;
    int count = 10;
    for (auto i : range(count))
    {
        EXPECT_EQ(i, other_i);
        other_i++;
    }
    EXPECT_EQ(count, other_i);
}

TEST(Range, basic_lower_upper)
{
    int count = 10;
    int start = std::rand();
    int other_i = start;
    for (auto i : range(start, start+count))
    {
        EXPECT_EQ(i, other_i);
        other_i++;
    }
    EXPECT_EQ(start+count, other_i);
}

TEST(Range, zero)
{
    int actual_count = 0;
    int count = 0;
    for (auto i : range(count))
    {
        (void)i;
        actual_count++;
    }
    EXPECT_EQ(count, actual_count);
}

TEST(Range, identical_bounds)
{
    int actual_count = 0;
    int b = std::rand();
    for (auto i : range(b, b))
    {
        (void)i;
        actual_count++;
    }
    EXPECT_EQ(0, actual_count);
}
#endif
