/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2014  Soumyajit De
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/GPUMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/statistics/experimental/internals/mmd/UnbiasedIncomplete.h>

using namespace shogun;
using namespace internal;
using namespace mmd;

UnbiasedIncomplete::UnbiasedIncomplete(index_t _n) : n(_n)
{
}

float64_t UnbiasedIncomplete::operator()(SGMatrix<float64_t> km)
{
	using MatrixXt = const Eigen::MatrixXd;
	using Block = const Eigen::Block<Eigen::Map<MatrixXt>>;

	Eigen::Map<MatrixXt> map(km.matrix, km.num_rows, km.num_cols);

	Block& b_x = map.block(0, 0, n, n);
	auto term_1 = b_x.sum() - b_x.diagonal().sum();

	Block& b_y = map.block(n, n, n, n);
	auto term_2 = b_y.sum() - b_y.diagonal().sum();

	Block& b_xy = map.block(n, 0, n, n);
	auto term_3 = b_xy.sum() - b_xy.diagonal().sum();

	auto statistic = term_1/n/(n-1) + term_2/n/(n-1) - 2*term_3/n/n;

	return statistic;

}
