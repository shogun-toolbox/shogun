/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 *
 */
#include <algorithm>
#include <shogun/distance/LevensteinDistance.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

LevensteinDistance::LevensteinDistance()
{
}

LevensteinDistance::LevensteinDistance(
    const std::string& lhs, const std::string& rhs)
    : lhs_name(lhs), rhs_name(rhs)
{
}

LevensteinDistance::~LevensteinDistance()
{
}

size_t
LevensteinDistance::compute(const std::string& word1, const std::string& word2)
{
	require(word1.size() != 0, "Left hand side name cannot be empty!");
	require(word2.size() != 0, "Right hand side name cannot be empty!");

	int m = word1.size(), n = word2.size();
	std::vector<std::vector<size_t>> dist(m + 1, std::vector<size_t>(n + 1));
	for (int i = 0; i <= m; ++i)
		dist[i][0] = i;
	for (int i = 0; i <= n; ++i)
		dist[0][i] = i;
	for (int i = 1; i <= m; ++i)
	{
		for (int j = 1; j <= n; ++j)
		{
			if (word1[i - 1] == word2[j - 1])
			{
				dist[i][j] = dist[i - 1][j - 1];
			}
			else
			{
				dist[i][j] = std::min(
				                 dist[i - 1][j - 1],
				                 std::min(dist[i - 1][j], dist[i][j - 1])) +
				             1;
			}
		}
	}
	return dist[m][n];
}

size_t LevensteinDistance::distance()
{
	return compute(lhs_name, rhs_name);
}

size_t
LevensteinDistance::distance(const std::string& lhs, const std::string& rhs)
{
	return compute(lhs, rhs);
}
