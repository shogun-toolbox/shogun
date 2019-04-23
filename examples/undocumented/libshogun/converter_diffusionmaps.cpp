/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Pan Deng
 */

#include <shogun/lib/config.h>

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/DiffusionMaps.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();

	int N = 100;
	int dim = 3;
	float64_t* matrix = new double[N*dim];
	for (int i=0; i<N*dim; i++)
		matrix[i] = std::sin((i / float64_t(N * dim)) * 3.14);

	auto features = std::make_shared<DenseFeatures<double>>(SGMatrix<double>(matrix,dim,N));
	auto dmaps = std::make_shared<DiffusionMaps>();
	dmaps->set_target_dim(2);
	dmaps->set_t(10);
	dmaps->parallel->set_num_threads(4);
	auto embedding = dmaps->transform(features);
	exit_shogun();
	return 0;
}
