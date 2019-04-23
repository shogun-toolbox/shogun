/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Soeren Sonnenburg, Pan Deng
 */

#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/LaplacianEigenmaps.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

int main(int argc, char** argv)
{
	int N = 100;
	int dim = 3;
	float64_t* matrix = new double[N*dim];
	for (int i=0; i<N*dim; i++)
		matrix[i] = std::sin((i / float64_t(N * dim)) * 3.14);

	DenseFeatures<double>* features = new DenseFeatures<double>(SGMatrix<double>(matrix,dim,N));
	CLaplacianEigenmaps* lem = new CLaplacianEigenmaps();
	lem->set_target_dim(2);
	lem->set_k(10);
	lem->get_global_parallel()->set_num_threads(4);
	auto embedding = lem->transform(features);
	return 0;
}
#else //USE_GPL_SHOGUN
int main(int argc, char** argv)
{
	return 0;
}
#endif //USE_GPL_SHOGUN
