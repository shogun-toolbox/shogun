#include <labels/MulticlassLabels.h>
#include <features/DenseFeatures.h>
#include <multiclass/GaussianNaiveBayes.h>
#include <base/init.h>
#include <lib/common.h>
#include <io/SGIO.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	// create some data
	SGMatrix<float64_t> matrix(2,3);
	for (int32_t i=0; i<6; i++)
		matrix.matrix[i]=i;

	// create three 2-dimensional vectors
	// shogun will now own the matrix created
	CDenseFeatures<float64_t>* features= new CDenseFeatures<float64_t>(matrix);

	// create three labels
	CMulticlassLabels* labels=new CMulticlassLabels(3);
	labels->set_label(0, 0);
	labels->set_label(1, +1);
	labels->set_label(2, +2);

	CGaussianNaiveBayes* ci = new CGaussianNaiveBayes(features,labels);
	ci->train();

	// classify on training examples
	for (int32_t i=0; i<3; i++)
		SG_SPRINT("output[%d]=%f\n", i, ci->apply_one(i));

	// free up memory
	SG_UNREF(ci);

	exit_shogun();
	return 0;
}
