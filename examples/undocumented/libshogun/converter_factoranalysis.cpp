/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Pan Deng, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/FactorAnalysis.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun();

	int N = 100;
	int dim = 3;
	float64_t* matrix = new double[N*dim];
	for (int i=0; i<N*dim; i++)
		matrix[i] = CMath::sin((i/float64_t(N*dim))*3.14);

	CDenseFeatures<double>* features = new CDenseFeatures<double>(SGMatrix<double>(matrix,dim,N));
	SG_REF(features);
	CFactorAnalysis* fa = new CFactorAnalysis();
	CDenseFeatures<double>* embedding = fa->embed(features);
	SG_UNREF(embedding);
	SG_UNREF(fa);
	SG_UNREF(features);
	exit_shogun();
	return 0;
}
