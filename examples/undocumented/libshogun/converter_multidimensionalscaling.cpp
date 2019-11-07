/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Pan Deng
 */

#include <shogun/lib/config.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/MultidimensionalScaling.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

int main(int argc, char** argv)
{
	int N = 100;
	int dim = 3;
	float64_t* matrix = new double[N*dim];
	for (int i=0; i<N*dim; i++)
		matrix[i] = std::sin((i / float64_t(N * dim)) * 3.14);

	auto features = std::make_shared<DenseFeatures<double>>(SGMatrix<double>(matrix,dim,N));
	auto mds = std::make_shared<MultidimensionalScaling>();
	mds->set_target_dim(2);
	mds->set_landmark(true);
	mds->get_global_parallel()->set_num_threads(4);
	auto embedding = mds->transform(features);
	return 0;
}
