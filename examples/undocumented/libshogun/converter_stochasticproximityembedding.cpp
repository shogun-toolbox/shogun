/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Heiko Strathmann,
 *          Fernando Iglesias, Bjoern Esser, Pan Deng
 */

#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/StochasticProximityEmbedding.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

int main()
{
	int N = 100;
	int dim = 3;

	// Generate toy data
	SGMatrix< float64_t > matrix(dim, N);
	for (int i=0; i<N*dim; i++)
		matrix[i] = std::sin((i / float64_t(N * dim)) * 3.14);

	DenseFeatures< float64_t >* features = new DenseFeatures<float64_t>(matrix);

	// Create embedding and set parameters for global strategy
	CStochasticProximityEmbedding* spe = new CStochasticProximityEmbedding();
	spe->set_target_dim(2);
	spe->set_strategy(SPE_GLOBAL);
	spe->set_nupdates(40);

	// Apply embedding with global strategy
	auto embedding = spe->transform(features);

	// Set parameters for local strategy
	spe->set_strategy(SPE_LOCAL);
	spe->set_k(12);

	// Apply embedding with local strategy
	embedding = spe->transform(features);

	// Free memory

	return 0;
}
#else //USE_GPL_SHOGUN
int main(int argc, char** argv)
{
	return 0;
}
#endif //USE_GPL_SHOGUN
