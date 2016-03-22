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
 * along with this program.  If not, see <http:/www.gnu.org/licenses/>.
 */

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/GPUMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/statistical_testing/internals/mmd/FullDirect.h>

using namespace shogun;
using namespace internal;
using namespace mmd;

float64_t FullDirect::operator()(SGMatrix<float64_t> km)
{
	Eigen::Map<Eigen::MatrixXd> map(km.matrix, km.num_rows, km.num_cols);
	index_t B = km.num_rows;

	Eigen::VectorXd diag = map.diagonal();
	map.diagonal().setZero();

	auto term_1 = CMath::sq(map.array().sum()/B/(B-1));
	auto term_2 = map.array().square().sum()/B/(B-1);
	auto term_3 = (map.colwise().sum()/(B-1)).array().sum()/B;

	map.diagonal() = diag;

	auto variance_estimate = 2*(term_1 + term_2 - 2 * term_3);

	return variance_estimate;

}
