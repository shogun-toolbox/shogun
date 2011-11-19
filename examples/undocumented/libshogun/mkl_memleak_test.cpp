#include <shogun/base/init.h>
#include <shogun/regression/svr/MKLRegression.h>
#include <shogun/features/Labels.h>
#include <shogun/lib/DataType.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/features/SimpleFeatures.h>

using namespace shogun;

#define N 1000

int main(int argc, char** argv)
{
	init_shogun();

	CMKLRegression* regressor = new CMKLRegression();

	SGVector<float64_t> vec(N);
	for (int i=0; i<N; i++) vec.vector[i] = i>>3;
	CLabels* labels = new CLabels(vec);

	SGMatrix<float64_t> matrix1(N,N);
	SGMatrix<float64_t> matrix2(N,N);

	for (int i=0; i<N; i++)
		for (int j=0; j<N; j++)
		{
			matrix1.matrix[i*N+j] = (i-j)*(i-j)/(1000.0*1000.0);
			matrix2.matrix[i*N+j] = (i-j)*(i-j)/(1000.0*1000.0);
		}

	CCustomKernel* kernel1 = new CCustomKernel(matrix1);
	CCustomKernel* kernel2 = new CCustomKernel(matrix2);

	matrix1.destroy_matrix();
	matrix2.destroy_matrix();

	CCombinedKernel* ckernel = new CCombinedKernel();
	ckernel->append_kernel(kernel1);
	ckernel->append_kernel(kernel2);

	regressor->set_kernel(ckernel);
	regressor->set_labels(labels);
	regressor->train();
	regressor->apply();

	SG_UNREF(ckernel);
	SG_UNREF(regressor);
	SG_UNREF(labels);
	exit_shogun();
	return 0;
}
