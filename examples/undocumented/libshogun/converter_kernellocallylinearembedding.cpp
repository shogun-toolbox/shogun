/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Soeren Sonnenburg, Pan Deng
 */

#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/KernelLocallyLinearEmbedding.h>
#include <shogun/kernel/LinearKernel.h>
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

	CDenseFeatures<double>* features = new CDenseFeatures<double>(SGMatrix<double>(matrix,dim,N));
	SG_REF(features);
	CKernelLocallyLinearEmbedding* klle = new CKernelLocallyLinearEmbedding();
	CKernel* kernel = new CLinearKernel();
	klle->set_target_dim(2);
	klle->set_k(4);
	klle->set_kernel(kernel);
	klle->parallel->set_num_threads(4);
	CDenseFeatures<double>* embedding = klle->embed(features);
	SG_UNREF(embedding);
	SG_UNREF(klle);
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
