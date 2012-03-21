#include <shogun/features/SimpleFeatures.h>
#include <shogun/kernel/GaussianKernel.h>

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
	float64_t* matrix = SG_MALLOC(float64_t, 6);
	
	//Labels
	SGVector<float64_t> labels(3);
	
	//First an Identity Matrix, later used for intermediate computations
	SGMatrix<float64_t> temp2(3,3);
	
	//Matrix for intermediate computations
	SGMatrix<float64_t> temp1(3,3);
	
	//Predictions
	SGVector<float64_t> result(3);

	labels[0] = -1;
	labels[1] = 1;
	labels[2] = -1;
	
	temp2[0] = 1;
	temp2[1] = 0;
	temp2[2] = 0;
	temp2[3] = 0;
	temp2[4] = 1;
	temp2[5] = 0;
	temp2[6] = 0;
	temp2[7] = 0;
	temp2[8] = 1;
	
	temp1[0] = 1;
	temp1[1] = 0;
	temp1[2] = 0;
	temp1[3] = 0;
	temp1[4] = 1;
	temp1[5] = 0;
	temp1[6] = 0;
	temp1[7] = 0;
	temp1[8] = 1;

	for (int32_t i=0; i<6; i++)
		matrix[i]=i;

	// create three 2-dimensional vectors 
	// shogun will now own the matrix created
	
	//Training Features
	CSimpleFeatures<float64_t>* features= new CSimpleFeatures<float64_t>();
	
	//Testing Features
	CSimpleFeatures<float64_t>* features_test= new CSimpleFeatures<float64_t>();
	features->set_feature_matrix(matrix, 2, 3);
	features_test->set_feature_matrix(matrix, 2, 3);

	
	// create gaussian kernels with cache 10MB, width 0.5
	CGaussianKernel* kernel = new CGaussianKernel(10, 0.5);
	kernel->init(features, features);

	CGaussianKernel* kernel_test = new CGaussianKernel(10, 0.5);
	kernel_test->init(features_test, features);
	
	/* We wish to calculate K(X_test, X_train)*(K(X_train, X_train)+sigma^(2)*I)^-1 * labels
	 * for mean predictions. In this case, sigma = 1
	 */
	
	//Calculate first (K(X_train, X_train)+sigma*I)
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, kernel->get_kernel_matrix().num_rows, 
		    temp2.num_cols, kernel->get_kernel_matrix().num_cols, 1.0,
		    kernel->get_kernel_matrix().matrix, kernel->get_kernel_matrix().num_cols, 
		    temp2.matrix, temp2.num_cols, 1.0, temp1.matrix, temp1.num_cols);
	
	//Take inverse of (K(X_train, X_train)+sigma*I)
	CMath::inverse(temp1);
	
	//Then multiply K(X_test, X_train) by (K(X_train, X_train) + sigma*I)^-1)
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, kernel_test->get_kernel_matrix().num_rows, 
		    temp1.num_cols, kernel_test->get_kernel_matrix().num_cols, 1.0,
		    kernel_test->get_kernel_matrix().matrix, kernel_test->get_kernel_matrix().num_cols, 
		    temp1.matrix, temp1.num_cols, 0.0, temp2.matrix, temp2.num_cols);
	
	//Finally multiply result by labels to obtain mean predictions on training
	//examples
	CMath::dgemv(1.0, temp1.matrix, temp1.num_rows, 3,
			CblasNoTrans, labels.vector, 0.0,
			result.vector);
	
	// output predictions
	for (int32_t i=0; i<3; i++)
		SG_SPRINT("output[%d]=%f\n", i, result[i]);

	// free up memory
	SG_FREE(kernel);
	SG_FREE(kernel_test);
	SG_FREE(features);
	SG_FREE(features_test);
	SG_FREE(matrix);
	temp2.destroy_matrix();
	temp1.destroy_matrix();
	result.destroy_vector();
	labels.destroy_vector();
			
	#endif
	exit_shogun();
	return 0;
}


	
