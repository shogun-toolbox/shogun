#include <shogun/base/init.h>
#include <shogun/features/Labels.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/classifier/svm/NewtonSVM.h>

#include <string.h>
using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc,char *argv[])
{
	init_shogun(&print_message,&print_message,&print_message);//initialising shogun without giving arguments shogun wont be able to print
	int32_t x_n=5,x_d=3;//X dimensions : x_n for no of datapoints and x_d for dimensionality of data
	SGMatrix<float64_t> fmatrix(x_d,x_n);



/*Initialising Feature Matrix */

	for (int i=0; i<x_n*x_d; i++)
		fmatrix.matrix[i] = i+1;
	fmatrix.matrix[2]=20;
//	for (int32_t i=0; i<x_n*x_d; i++)
//		fmatrix.matrix[i]=CMath::randn_doubl();		
	//CMath::transpose_matrix(fmatrix.matrix,x_d,x_n);
	SG_SPRINT("FEATURE MATRIX :\n");	
	CMath::display_matrix(fmatrix.matrix,x_n,x_d);

	CSimpleFeatures<float64_t>* features = new CSimpleFeatures<float64_t>(fmatrix);
	SG_REF(features);
	
/*Creating random labels */
	CLabels* labels=new CLabels(x_n);
	
	// create labels, two classes 
	//for (int32_t i=0; i<x_n; ++i)
	//	labels->set_label(i, i%2==0 ? 1 : -1);
	
	labels->set_label(0,1);
	labels->set_label(1,-1);
	labels->set_label(2,1);
	labels->set_label(3,1);
	labels->set_label(4,-1);
	SG_REF(labels);
	
/*Working withNewton SVM */

	float64_t lambda=1;
	int32_t iter=20;	

	CNewtonSVM *nsvm = new CNewtonSVM(lambda,iter,features,labels);
	SG_REF(nsvm);
	nsvm->train();
	exit_shogun();
	return 0;
}


