/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Heiko Strathmann, 
 *          Fernando Iglesias, Björn Esser, Pan Deng
 */

#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/StochasticProximityEmbedding.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

int main()
{
	init_shogun_with_defaults();

	int N = 100;
	int dim = 3;

	// Generate toy data
	SGMatrix< float64_t > matrix(dim, N);
	for (int i=0; i<N*dim; i++)
		matrix[i] = CMath::sin((i/float64_t(N*dim))*3.14);

	CDenseFeatures< float64_t >* features = new CDenseFeatures<float64_t>(matrix);
	SG_REF(features);

	// Create embedding and set parameters for global strategy
	CStochasticProximityEmbedding* spe = new CStochasticProximityEmbedding();
	spe->set_target_dim(2);
	spe->set_strategy(SPE_GLOBAL);
	spe->set_nupdates(40);
	SG_REF(spe);

	// Apply embedding with global strategy
	CDenseFeatures< float64_t >* embedding = spe->embed(features);
	SG_REF(embedding);

	// Set parameters for local strategy
	spe->set_strategy(SPE_LOCAL);
	spe->set_k(12);

	// Apply embedding with local strategy
	SG_UNREF(embedding);
	embedding = spe->embed(features);
	SG_REF(embedding);

	// Free memory
	SG_UNREF(embedding);
	SG_UNREF(spe);
	SG_UNREF(features);

	exit_shogun();

	return 0;
}
#else //USE_GPL_SHOGUN
int main(int argc, char** argv)
{
	return 0;
}
#endif //USE_GPL_SHOGUN
