#include <shogun/features/Labels.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/multiclass/MulticlassLibSVM.h>
#include <shogun/base/init.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char** argv)
{
	init_shogun(&print_message);
	index_t num_vec=3;
	index_t num_feat=2;
	index_t num_class=2;

	// create some data
	SGMatrix<float64_t> matrix(num_feat, num_vec);
	CMath::range_fill_vector(matrix.matrix, num_feat*num_vec);

	// create vectors
	// shogun will now own the matrix created
	CSimpleFeatures<float64_t>* features=new CSimpleFeatures<float64_t>(matrix);

	// create three labels
	CLabels* labels=new CLabels(num_vec);
	for (index_t i=0; i<num_vec; ++i)
		labels->set_label(i, i%num_class);

	// create gaussian kernel with cache 10MB, width 0.5
	CGaussianKernel* kernel = new CGaussianKernel(10, 0.5);
	kernel->init(features, features);

	// create libsvm with C=10 and train
	CMulticlassLibSVM* svm = new CMulticlassLibSVM(10, kernel, labels);
	svm->train();

	// classify on training examples
	CLabels* output=svm->apply();
	CMath::display_vector(output->get_labels().vector, output->get_num_labels(),
			"batch output");

	/* assert that batch apply and apply(index_t) give same result */
	for (index_t i=0; i<output->get_num_labels(); ++i)
	{
		float64_t label=svm->apply(i);
		SG_SPRINT("single output[%d]=%f\n", i, label);
		ASSERT(output->get_label(i)==label);
	}
	SG_UNREF(output);

	// free up memory
	SG_UNREF(svm);

	exit_shogun();
	return 0;
}

