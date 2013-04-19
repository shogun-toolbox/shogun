#include <shogun/io/AsciiFile.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/multiclass/MulticlassOneVsOneStrategy.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/base/init.h>

#define  EPSILON  1e-5

using namespace shogun;

/* file data */
const char fname_feats[]="../data/fm_train_real.dat";
const char fname_labels[]="../data/label_train_multiclass.dat";

void test()
{
	/* dense features from matrix */
	CAsciiFile* feature_file = new CAsciiFile(fname_feats);
	SGMatrix<float64_t> mat=SGMatrix<float64_t>();
	mat.load(feature_file);
	SG_UNREF(feature_file);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(mat);
	SG_REF(features);

	/* labels from vector */
	CAsciiFile* label_file = new CAsciiFile(fname_labels);
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
	CLinearMulticlassMachine* mc_svm_ova = new CLinearMulticlassMachine(
			new CMulticlassOneVsRestStrategy(), (CDotFeatures*) features, svm, labels);
	SG_REF(mc_svm_ova);

	// Train the multiclass machine using the data passed in the constructor
	mc_svm_ova->train();

	// Classify the training examples and show the results
    SG_SPRINT("-- OVA: Use direct SVM outputs --\n");
	CMulticlassLabels* output_ova = CMulticlassLabels::obtain_from_generic(mc_svm_ova->apply());

	SGVector< int32_t > out_labels = output_ova->get_int_labels();
	SGVector<int32_t>::display_vector(out_labels.vector, out_labels.vlen);

    for (int32_t i=0; i<output_ova->get_num_labels(); i++)
    {
        SG_SPRINT("out_values[%d] = ", i);
        SGVector<float64_t> out_values = output_ova->get_multiclass_confidences(i);
	    SGVector<float64_t>::display_vector(out_values.vector, out_values.vlen);
        SG_SPRINT("\n");
    }

    SG_SPRINT("-- OVA: Use probabilistic SVM outputs --\n");
    mc_svm_ova->set_prob_support();
	CMulticlassLabels* output_p_ova = CMulticlassLabels::obtain_from_generic(mc_svm_ova->apply());

	out_labels = output_p_ova->get_int_labels();
	SGVector<int32_t>::display_vector(out_labels.vector, out_labels.vlen);

    for (int32_t i=0; i<output_p_ova->get_num_labels(); i++)
    {
        SG_SPRINT("out_values[%d] = ", i);
        SGVector<float64_t> out_values = output_p_ova->get_multiclass_confidences(i);
	    SGVector<float64_t>::display_vector(out_values.vector, out_values.vlen);
        SG_SPRINT("\n");
    }

	// Create a multiclass svm classifier that consists of several of the previous one
	CLinearMulticlassMachine* mc_svm_ovo = new CLinearMulticlassMachine(
			new CMulticlassOneVsOneStrategy(), (CDotFeatures*) features, svm, labels);
	SG_REF(mc_svm_ovo);

	// Train the multiclass machine using the data passed in the constructor
	mc_svm_ovo->train();

	// Classify the training examples and show the results
    SG_SPRINT("-- OVO: Use direct SVM outputs --\n");
	CMulticlassLabels* output_ovo = CMulticlassLabels::obtain_from_generic(mc_svm_ovo->apply());

	out_labels = output_ovo->get_int_labels();
	SGVector<int32_t>::display_vector(out_labels.vector, out_labels.vlen);

    for (int32_t i=0; i<output_ovo->get_num_labels(); i++)
    {
        SG_SPRINT("out_values[%d] = ", i);
        SGVector<float64_t> out_values = output_ovo->get_multiclass_confidences(i);
	    SGVector<float64_t>::display_vector(out_values.vector, out_values.vlen);
        SG_SPRINT("\n");
    }

    SG_SPRINT("-- OVO: Use probabilistic SVM outputs --\n");
    mc_svm_ovo->set_prob_support();
	CMulticlassLabels* output_p_ovo = CMulticlassLabels::obtain_from_generic(mc_svm_ovo->apply());

	out_labels = output_p_ovo->get_int_labels();
	SGVector<int32_t>::display_vector(out_labels.vector, out_labels.vlen);

    for (int32_t i=0; i<output_p_ovo->get_num_labels(); i++)
    {
        SG_SPRINT("out_values[%d] = ", i);
        SGVector<float64_t> out_values = output_p_ovo->get_multiclass_confidences(i);
	    SGVector<float64_t>::display_vector(out_values.vector, out_values.vlen);
        SG_SPRINT("\n");
    }

	//Free resources
	SG_UNREF(mc_svm_ova);
	SG_UNREF(mc_svm_ovo);
	SG_UNREF(svm);
	SG_UNREF(output_ova);
	SG_UNREF(output_ovo);
	SG_UNREF(output_p_ova);
	SG_UNREF(output_p_ovo);
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

