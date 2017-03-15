#include <cmath>
#include <thread>
#include <chrono>
#include <shogun/io/SGIO.h>

#include <gtest/gtest.h>

using namespace shogun;

const int millis= 10;

TEST(SGIOTest, progress_correct_bounds_positive)
{
	SGIO tmp;
	tmp.enable_progress();
	for (int i=0; i<100; i++)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(millis));
		tmp.progress(i, 0, 100);
		EXPECT_EQ(std::ceil(tmp.get_last_progress()), i+1);
	}
}

TEST(SGIOTest, progress_correct_bounds_negative)
{
	SGIO tmp2;
	tmp2.enable_progress();
	for (int i=-100; i<0; i++)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(millis));
		tmp2.progress(i, -100, 0);
		EXPECT_EQ(std::ceil(tmp2.get_last_progress()), (100+i)+1);
	}
}

TEST(SGIOTest, progress_incorrect_bounds_positive)
{
	SGIO tmp2;
	tmp2.enable_progress();
	tmp2.progress(0, 100, 1);
	EXPECT_FLOAT_EQ(tmp2.get_last_progress(), (float64_t)0);
}

TEST(SGIOTest, progress_incorrect_bounds_negative)
{
	SGIO tmp;
	tmp.enable_progress();
	tmp.progress(0, -1, -2);
	EXPECT_FLOAT_EQ(tmp.get_last_progress(), (float64_t)0);
}

TEST(SGIOTest, progress_incorrect_bounds_equal)
{
	SGIO tmp3;
	tmp3.enable_progress();
	tmp3.progress(0, 1, 1);
	EXPECT_FLOAT_EQ(tmp3.get_last_progress(), (float64_t)0);
}

TEST(SGIOTest, progress_current_val_out_of_bounds_lower)
{
	SGIO tmp;
	tmp.enable_progress();
	tmp.progress(-1, 0, 100);
	EXPECT_FLOAT_EQ(tmp.get_last_progress(), (float64_t)1e-5);
}

TEST(SGIOTest, progress_current_val_out_of_bounds_higher)
{
	SGIO tmp2;
	tmp2.enable_progress();
	tmp2.progress(1001, 0, 100);
	EXPECT_FLOAT_EQ(tmp2.get_last_progress(), (float64_t)100);
}
