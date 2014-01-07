#include <base/init.h>
#include <features/Labels.h>
#include <features/DenseFeatures.h>
#include <mathematics/Math.h>
#include <classifier/svm/NewtonSVM.h>


using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc,char *argv[])
{
	init_shogun(&print_message,&print_message,&print_message);//initialising shogun without giving arguments shogun wont be able to print
	int32_t x_n=4,x_d=2;//X dimensions : x_n for no of datapoints and x_d for dimensionality of data
	SGMatrix<float64_t> fmatrix(x_d,x_n);


	SG_SPRINT("\nTEST 1:\n\n");

/*Initialising Feature Matrix */

	for (int i=0; i<x_n*x_d; i++)
		fmatrix.matrix[i] = i+1;
	SG_SPRINT("FEATURE MATRIX :\n");
	CMath::display_matrix(fmatrix.matrix,x_d,x_n);

	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(fmatrix);
	SG_REF(features);

/*Creating random labels */
	CLabels* labels=new CLabels(x_n);

	// create labels, two classes
	labels->set_label(0,1);
	labels->set_label(1,-1);
	labels->set_label(2,1);
	labels->set_label(3,1);
	SG_REF(labels);

/*Working with Newton SVM */

	float64_t lambda=1.0;
	int32_t iter=20;

	CNewtonSVM *nsvm = new CNewtonSVM(lambda,features,labels,iter);
	SG_REF(nsvm);
	nsvm->train();
	SG_UNREF(labels);
	SG_UNREF(nsvm);

	SG_SPRINT("TEST 2:\n\n");


	x_n=5;
	x_d=3;
	SGMatrix<float64_t> fmatrix2(x_d,x_n);
	for (int i=0; i<x_n*x_d; i++)
		fmatrix2.matrix[i] = i+1;
	SG_SPRINT("FEATURE MATRIX :\n");
	CMath::display_matrix(fmatrix2.matrix,x_d,x_n);
	features->set_feature_matrix(fmatrix2);
	SG_REF(features);

/*Creating random labels */
	CLabels* labels2=new CLabels(x_n);

	// create labels, two classes
	labels2->set_label(0,1);
	labels2->set_label(1,-1);
	labels2->set_label(2,1);
	labels2->set_label(3,1);
	labels2->set_label(4,-1);
	SG_REF(labels2);

/*Working with Newton SVM */

	lambda=1.0;
	iter=20;

	CNewtonSVM *nsvm2 = new CNewtonSVM(lambda,features,labels2,iter);
	SG_REF(nsvm2);
	nsvm2->train();


	SG_UNREF(labels2);
	SG_UNREF(nsvm2);
	SG_UNREF(features);
	exit_shogun();
	return 0;
}


