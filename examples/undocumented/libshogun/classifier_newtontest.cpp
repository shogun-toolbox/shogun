#include <shogun/features/Labels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/classifier/svm/NewtonSVM.h>


using namespace shogun;

int main(int argc,char *argv[])
{
	int32_t x_n=4,x_d=2;//X dimensions : x_n for no of datapoints and x_d for dimensionality of data
	SGMatrix<float64_t> fmatrix(x_d,x_n);


	SG_SPRINT("\nTEST 1:\n\n");

/*Initialising Feature Matrix */

	for (int i=0; i<x_n*x_d; i++)
		fmatrix.matrix[i] = i+1;
	SG_SPRINT("FEATURE MATRIX :\n");
	Math::display_matrix(fmatrix.matrix,x_d,x_n);

	DenseFeatures<float64_t>* features = new DenseFeatures<float64_t>(fmatrix);

/*Creating random labels */
	Labels* labels=new Labels(x_n);

	// create labels, two classes
	labels->set_label(0,1);
	labels->set_label(1,-1);
	labels->set_label(2,1);
	labels->set_label(3,1);

/*Working with Newton SVM */

	float64_t lambda=1.0;
	int32_t iter=20;

	CNewtonSVM *nsvm = new CNewtonSVM(lambda,features,labels,iter);
	nsvm->train();

	SG_SPRINT("TEST 2:\n\n");


	x_n=5;
	x_d=3;
	SGMatrix<float64_t> fmatrix2(x_d,x_n);
	for (int i=0; i<x_n*x_d; i++)
		fmatrix2.matrix[i] = i+1;
	SG_SPRINT("FEATURE MATRIX :\n");
	Math::display_matrix(fmatrix2.matrix,x_d,x_n);
	features->set_feature_matrix(fmatrix2);

/*Creating random labels */
	Labels* labels2=new Labels(x_n);

	// create labels, two classes
	labels2->set_label(0,1);
	labels2->set_label(1,-1);
	labels2->set_label(2,1);
	labels2->set_label(3,1);
	labels2->set_label(4,-1);

/*Working with Newton SVM */

	lambda=1.0;
	iter=20;

	CNewtonSVM *nsvm2 = new CNewtonSVM(lambda,features,labels2,iter);
	nsvm2->train();

	return 0;
}


