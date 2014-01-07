#include <io/CSVFile.h>
#include <labels/MulticlassLabels.h>
#include <io/SGIO.h>
#include <features/DenseFeatures.h>
#include <multiclass/MulticlassStrategy.h>
#include <multiclass/MulticlassOneVsOneStrategy.h>
#include <multiclass/MulticlassOneVsRestStrategy.h>
#include <machine/LinearMulticlassMachine.h>
#include <classifier/svm/LibLinear.h>
#include <base/init.h>

#define  EPSILON  1e-5

using namespace shogun;

/* file data */
const char fname_feats[]="../data/fm_train_real.dat";
const char fname_labels[]="../data/label_train_multiclass.dat";

void test()
{
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

	// Create liblinear svm classifier with L2-regularized L2-loss
	CLibLinear* svm = new CLibLinear(L2R_L2LOSS_SVC);
	SG_REF(svm);

	// Add some configuration to the svm
	svm->set_epsilon(EPSILON);
	svm->set_bias_enabled(true);

	// Create a multiclass svm classifier that consists of several of the previous one
	// There are several heuristics are implemented:
	// OVA_NORM, OVA_SOFTMAX
	// OVO_PRICE, OVO_HASTIE, OVO_HAMAMURA
	CLinearMulticlassMachine* mc_svm = new CLinearMulticlassMachine(
			new CMulticlassOneVsOneStrategy(OVO_HASTIE), (CDotFeatures*) features, svm, labels);
	SG_REF(mc_svm);

	// Train the multiclass machine using the data passed in the constructor
	mc_svm->train();

	// Classify the training examples and show the results
	CMulticlassLabels* output = CLabelsFactory::to_multiclass(mc_svm->apply());

	SGVector< int32_t > out_labels = output->get_int_labels();
	SGVector<int32_t>::display_vector(out_labels.vector, out_labels.vlen);

	for (int32_t i=0; i<output->get_num_labels(); i++)
	{
		SG_SPRINT("out_values[%d] = ", i);
		SGVector<float64_t> out_values = output->get_multiclass_confidences(i);
		SGVector<float64_t>::display_vector(out_values.vector, out_values.vlen);
		SG_SPRINT("\n");
	}

	//Free resources
	SG_UNREF(mc_svm);
	SG_UNREF(svm);
	SG_UNREF(output);
	SG_UNREF(features);
	SG_UNREF(labels);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	//sg_io->set_loglevel(MSG_DEBUG);

	test();

	exit_shogun();

	return 0;
}

