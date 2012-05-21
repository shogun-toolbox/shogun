#include <shogun/features/DenseFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/labels/RegressionLabels.h>

/* Example mean prediction from a Gaussian Kernel adapted from 
 * classifier_minimal_svm.cpp
 * Jacob Walker
 */

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char** argv)
{
	init_shogun(&print_message);
	
#ifdef HAVE_LAPACK
	
	// create some data
	SGMatrix<float64_t> matrix(2,3);
	for (int32_t i=0; i<6; i++)
		matrix.matrix[i]=i;
	
	//Labels
	CRegressionLabels* labels = new CRegressionLabels(3);
	SG_REF(labels);
	
	labels->set_label(0, -1);
	labels->set_label(1, +1);
	labels->set_label(2, -1);
	
	// create three 2-dimensional vectors 
	// shogun will now own the matrix created
	CDenseFeatures<float64_t>* features= new CDenseFeatures<float64_t>(matrix);
	SG_REF(features);
	
	// create gaussian kernel with cache 10MB, width 0.5
	CGaussianKernel* kernel = new CGaussianKernel(10, 0.5);
	SG_REF(kernel);
	
	//Gaussian Process Regression with sigma = 1.
	CGaussianProcessRegression regressor(1.0, kernel, features, labels);
	
	regressor.train(features);
	//Get mean predictions
	CRegressionLabels* result = CRegressionLabels::obtain_from_generic(regressor.apply());
	SG_REF(result);
	
	SGMatrix<float64_t> cov = regressor.getCovarianceMatrix(features);
	
	CMath::display_matrix(cov.matrix, cov.num_rows, cov.num_cols, "Covariance Matrix");

	// output predictions
	for (int32_t i=0; i<3; i++)
		SG_SPRINT("output[%d]=%f\n", i, result->get_label(i));

	// free up memory
	SG_UNREF(result);
	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(kernel);
#endif
	
	exit_shogun();
	return 0;
}
