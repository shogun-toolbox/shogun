#include <shogun/io/CSVFile.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubsetFeatures.h>
#include <shogun/multiclass/ShareBoost.h>

#define  EPSILON  1e-5

using namespace shogun;

const char* fname_feats = "../data/7class_example4_train.dense";
const char* fname_labels = "../data/7class_example4_train.label";

int main(int argc, char** argv)
{
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

	SG_SPRINT("Performing ShareBoost on a %d-class problem\n", labels->get_num_classes());

	// Create ShareBoost Machine
	CShareBoost *machine = new CShareBoost(features, labels, 10);

	machine->train();

	SGVector<int32_t> activeset = machine->get_activeset();
	SG_SPRINT("%d out of %d features are selected:\n", activeset.vlen, mat.num_rows);
	for (int32_t i=0; i < activeset.vlen; ++i)
		SG_SPRINT("activeset[%02d] = %d\n", i, activeset[i]);

	CDenseSubsetFeatures<float64_t> *subset_fea = new CDenseSubsetFeatures<float64_t>(features, machine->get_activeset());
	MulticlassLabels* output =
	    machine->apply(subset_fea)->as<MulticlassLabels>();

	int32_t correct = 0;
	for (int32_t i=0; i < output->get_num_labels(); ++i)
		if (output->get_int_label(i) == labels->get_int_label(i))
			correct++;
	SG_SPRINT("Accuracy = %.4f\n", float64_t(correct)/labels->get_num_labels());

	return 0;
}
