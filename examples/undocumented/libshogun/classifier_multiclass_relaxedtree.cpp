#include <shogun/io/CSVFile.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubsetFeatures.h>
#include <shogun/base/init.h>
#include <shogun/multiclass/tree/RelaxedTree.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/kernel/GaussianKernel.h>

#define  EPSILON  1e-5

using namespace shogun;

const char* fname_feats = "../data/7class_example4_train.dense";
const char* fname_labels = "../data/7class_example4_train.label";

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	/* dense features from matrix */
	CSVFile* feature_file = new CSVFile(fname_feats);
	SGMatrix<float64_t> mat=SGMatrix<float64_t>();
	mat.load(feature_file);

	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t>(mat);

	/* labels from vector */
	CSVFile* label_file = new CSVFile(fname_labels);
	SGVector<float64_t> label_vec;
	label_vec.load(label_file);

	MulticlassLabels* labels=new MulticlassLabels(label_vec);

	// Create RelaxedTree Machine
	CRelaxedTree *machine = new CRelaxedTree();
	machine->set_labels(labels);
	Kernel *kernel = new GaussianKernel();
	machine->set_kernel(kernel);

	MulticlassLibLinear *svm = new MulticlassLibLinear();

	machine->set_machine_for_confusion_matrix(svm);
	machine->train(features);

	MulticlassLabels* output = machine->apply()->as<MulticlassLabels>();

	MulticlassAccuracy *evaluator = new MulticlassAccuracy();
	SG_SPRINT("Accuracy = %.4f\n", evaluator->evaluate(output, labels));

	// Free resources

	exit_shogun();

	return 0;
}

