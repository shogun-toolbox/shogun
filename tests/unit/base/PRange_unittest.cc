#include <cmath>
#include <gtest/gtest.h>
#include <shogun/base/progress.h>
#include <shogun/io/SGIO.h>
#include <thread>

using namespace shogun;

SGIO range_io;
const int millis = 10;
auto range_test = progress(range(0, 100), range_io);

TEST(PRange, basic_upper)
{
	int other_i = 0;
	int count = 10;
	for (auto i : progress(range(count), range_io))
	{
		EXPECT_EQ(i, other_i);
		other_i++;
	}
	EXPECT_EQ(count, other_i);
}

TEST(PRange, basic_lower_upper)
{
	int count = 10;
	int start = std::rand();
	int other_i = start;
	for (auto i : progress(range(start, start + count), range_io))
	{
		EXPECT_EQ(i, other_i);
		other_i++;
	}
	EXPECT_EQ(start + count, other_i);
}

TEST(PRange, zero)
{
	int actual_count = 0;
	int count = 0;
	for (auto i : progress(range(count), range_io))
	{
		(void)i;
		actual_count++;
	}
	EXPECT_EQ(count, actual_count);
}

TEST(PRange, identical_bounds)
{
	int actual_count = 0;
	int b = std::rand();
	for (auto i : progress(range(b, b), range_io))
	{
		(void)i;
		actual_count++;
	}
	EXPECT_EQ(0, actual_count);
}

TEST(PRange, progress_correct_bounds_positive)
{
	range_io.enable_progress();
	range_test = progress(range(0, 100), range_io);
	for (auto i : range_test)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(millis));
		EXPECT_EQ(std::ceil(range_test.get_last_progress()), i);
	}
}

TEST(PRange, progress_correct_bounds_negative)
{
	range_io.enable_progress();
	range_test = progress(range(-100, 0), range_io);
	for (auto i : range_test)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(millis));
		EXPECT_EQ(std::ceil(range_test.get_last_progress()), (100 + i));
	}
}

TEST(PRange, DISABLED_progress_incorrect_bounds_positive)
{
	range_io.enable_progress();
	range_test = progress(range(100, 0), range_io);
	for (auto i : range_test)
	{
		(void)i;
		std::this_thread::sleep_for(std::chrono::milliseconds(millis));
	}
	EXPECT_FLOAT_EQ(range_test.get_last_progress(), (float64_t)0);
}

TEST(PRange, DISABLED_progress_incorrect_bounds_negative)
{
	range_io.enable_progress();
	range_test = progress(range(100, 0), range_io);
	for (auto i : range_test)
	{
		(void)i;
		std::this_thread::sleep_for(std::chrono::milliseconds(millis));
	}
	EXPECT_FLOAT_EQ(range_test.get_last_progress(), (float64_t)0);
}

TEST(PRange, DISABLED_progress_incorrect_bounds_equal)
{
	range_io.enable_progress();
	range_test = progress(range(1, 1), range_io);
	for (auto i : range_test)
	{
		(void)i;
		std::this_thread::sleep_for(std::chrono::milliseconds(millis));
	}
	EXPECT_FLOAT_EQ(range_test.get_last_progress(), (float64_t)0);
}
