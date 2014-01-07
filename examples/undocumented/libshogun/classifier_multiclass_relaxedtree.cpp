#include <io/CSVFile.h>
#include <labels/MulticlassLabels.h>
#include <io/SGIO.h>
#include <features/DenseFeatures.h>
#include <features/DenseSubsetFeatures.h>
#include <base/init.h>
#include <multiclass/tree/RelaxedTree.h>
#include <multiclass/MulticlassLibLinear.h>
#include <evaluation/MulticlassAccuracy.h>
#include <kernel/GaussianKernel.h>

#define  EPSILON  1e-5

using namespace shogun;

const char* fname_feats = "../data/7class_example4_train.dense";
const char* fname_labels = "../data/7class_example4_train.label";

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	/* dense features from matrix */
	CCSVFile* feature_file = new CCSVFile(fname_feats);
	SGMatrix<float64_t> mat=SGMatrix<float64_t>();
	mat.load(feature_file);
	SG_UNREF(feature_file);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(mat);
	SG_REF(features);

	/* labels from vector */
	CCSVFile* label_file = new CCSVFile(fname_labels);
	SGVector<float64_t> label_vec;
	label_vec.load(label_file);
	SG_UNREF(label_file);

	CMulticlassLabels* labels=new CMulticlassLabels(label_vec);
	SG_REF(labels);

	// Create RelaxedTree Machine
	CRelaxedTree *machine = new CRelaxedTree();
	SG_REF(machine);
	machine->set_labels(labels);
	CKernel *kernel = new CGaussianKernel();
	SG_REF(kernel);
	machine->set_kernel(kernel);

	CMulticlassLibLinear *svm = new CMulticlassLibLinear();

	machine->set_machine_for_confusion_matrix(svm);
	machine->train(features);


	CMulticlassLabels* output = CLabelsFactory::to_multiclass(machine->apply());

	CMulticlassAccuracy *evaluator = new CMulticlassAccuracy();
	SG_SPRINT("Accuracy = %.4f\n", evaluator->evaluate(output, labels));

	// Free resources
	SG_UNREF(machine);
	SG_UNREF(output);
	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(evaluator);
	SG_UNREF(kernel);

	exit_shogun();

	return 0;
}

