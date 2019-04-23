#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/multiclass/MulticlassLibSVM.h>
#include <shogun/base/init.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();
	index_t num_vec=3;
	index_t num_feat=2;
	index_t num_class=2;

	// create some data
	SGMatrix<float64_t> matrix(num_feat, num_vec);
	SGVector<float64_t>::range_fill_vector(matrix.matrix, num_feat*num_vec);

	// create vectors
	// shogun will now own the matrix created
	auto features=std::make_shared<DenseFeatures<float64_t>>(matrix);

	// create three labels
	auto labels=std::make_shared<MulticlassLabels>(num_vec);
	for (index_t i=0; i<num_vec; ++i)
		labels->set_label(i, i%num_class);

	// create gaussian kernel with cache 10MB, width 0.5
	auto kernel = std::make_shared<GaussianKernel>(10, 0.5);
	kernel->init(features, features);

	// create libsvm with C=10 and train
	auto svm = std::make_shared<MulticlassLibSVM>(10, kernel, labels);
	svm->train();

	// classify on training examples
	auto output = svm->apply()->as<MulticlassLabels>();
	SGVector<float64_t>::display_vector(output->get_labels().vector, output->get_num_labels(),
			"batch output");

	/* assert that batch apply and apply(index_t) give same result */
	for (index_t i=0; i<output->get_num_labels(); ++i)
	{
		float64_t label=svm->apply_one(i);
		SG_SPRINT("single output[%d]=%f\n", i, label);
		ASSERT(output->get_label(i)==label);
	}

	// free up memory

	exit_shogun();
	return 0;
}

