#include <shogun/io/CSVFile.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubsetFeatures.h>
#include <shogun/base/init.h>
#include <shogun/multiclass/ShareBoost.h>

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

	SG_SPRINT("Performing ShareBoost on a %d-class problem\n", labels->get_num_classes());

	// Create ShareBoost Machine
	CShareBoost *machine = new CShareBoost(features, labels, 10);
	SG_REF(machine);

	machine->train();

	SGVector<int32_t> activeset = machine->get_activeset();
	SG_SPRINT("%d out of %d features are selected:\n", activeset.vlen, mat.num_rows);
	for (int32_t i=0; i < activeset.vlen; ++i)
		SG_SPRINT("activeset[%02d] = %d\n", i, activeset[i]);

	CDenseSubsetFeatures<float64_t> *subset_fea = new CDenseSubsetFeatures<float64_t>(features, machine->get_activeset());
	SG_REF(subset_fea);
	CMulticlassLabels* output = CLabelsFactory::to_multiclass(machine->apply(subset_fea));

	int32_t correct = 0;
	for (int32_t i=0; i < output->get_num_labels(); ++i)
		if (output->get_int_label(i) == labels->get_int_label(i))
			correct++;
	SG_SPRINT("Accuracy = %.4f\n", float64_t(correct)/labels->get_num_labels());

	// Free resources
	SG_UNREF(machine);
	SG_UNREF(output);
	SG_UNREF(subset_fea);
	SG_UNREF(features);
	SG_UNREF(labels);
	exit_shogun();

	return 0;
}
