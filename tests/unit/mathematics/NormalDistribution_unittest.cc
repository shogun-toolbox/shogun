#include <gtest/gtest.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <cmath>
#include <random>

using namespace shogun;

TEST(NormalDistribution, normal_distribution_test)
{
	const float64_t mean = 24;
	const float64_t stddev = 3;
	const int32_t num_samples = 5000;

	std::mt19937_64 prng(0);

	NormalDistribution<float64_t> dist{mean, stddev};
	SGVector<float64_t> samples(num_samples);

	int32_t count_stddev_away = 0;
	int32_t count_2stddev_away = 0;
	for (auto& sample : samples)
	{
		sample = dist(prng);
		if (std::abs(sample - mean) < stddev)
			count_stddev_away++;
		if (std::abs(sample - mean) < 2 * stddev)
			count_2stddev_away++;
	}

	float64_t percentage_stddev_away =
	    float64_t(count_stddev_away) / num_samples;

	float64_t percentage_2stddev_away =
	    float64_t(count_2stddev_away) / num_samples;

	auto calculated_mean = linalg::mean(samples);
	auto calculated_stddev =
	    linalg::std_deviation(SGMatrix<float64_t>(samples), false)[0];

	EXPECT_NEAR(mean, calculated_mean, 0.05);
	EXPECT_NEAR(stddev, calculated_stddev, 0.05);
	EXPECT_NEAR(percentage_stddev_away, 0.6827, 0.05);
	EXPECT_NEAR(percentage_2stddev_away, 0.9545, 0.05);
}
