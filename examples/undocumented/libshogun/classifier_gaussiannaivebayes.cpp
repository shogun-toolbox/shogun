#include <shogun/features/Labels.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/classifier/GaussianNaiveBayes.h>
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	// create some data
	float64_t* matrix = SG_MALLOC(float64_t, 6);
	for (int32_t i=0; i<6; i++)
		matrix[i]=i;

	// create three 2-dimensional vectors 
	// shogun will now own the matrix created
	CSimpleFeatures<float64_t>* features= new CSimpleFeatures<float64_t>();
	features->set_feature_matrix(matrix, 2, 3);

	// create three labels
	CLabels* labels=new CLabels(3);
	labels->set_label(0, 0);
	labels->set_label(1, +1);
	labels->set_label(2, +2);

	CGaussianNaiveBayes* ci = new CGaussianNaiveBayes(features,labels);
	ci->train();

	// classify on training examples
	for (int32_t i=0; i<3; i++)
		SG_SPRINT("output[%d]=%f\n", i, ci->apply(i));

	// free up memory
	SG_UNREF(ci);

	exit_shogun();
	return 0;
}
