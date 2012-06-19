#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/common.h>
#include <shogun/loss/HingeLoss.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/structure/MulticlassSOLabels.h>
#include <shogun/structure/MulticlassModel.h>
#include <shogun/structure/PrimalMosekSOSVM.h>

using namespace shogun;

#define	DIMS		2
#define EPSILON  	0
#define	NUM_SAMPLES	100
#define NUM_CLASSES	10

char FNAME[] = "data.out";

void gen_rand_data(SGVector< float64_t > labs, SGMatrix< float64_t > feats)
{
	float64_t means[DIMS];
	float64_t  stds[DIMS];

	FILE* pfile = fopen(FNAME, "w");

	for ( int32_t c = 0 ; c < NUM_CLASSES ; ++c )
	{
		for ( int32_t j = 0 ; j < DIMS ; ++j )
		{
			means[j] = CMath::random(-100, 100);
			 stds[j] = CMath::random(   1,   5);
		}

		for ( int32_t i = 0 ; i < NUM_SAMPLES ; ++i )
		{
			labs[c*NUM_SAMPLES+i] = c;
			
			fprintf(pfile, "%d", c);

			for ( int32_t j = 0 ; j < DIMS ; ++j )
			{
				feats[(c*NUM_SAMPLES+i)*DIMS + j] = 
					CMath::normal_random(means[j], stds[j]);

				fprintf(pfile, " %f", feats[(c*NUM_SAMPLES+i)*DIMS + j]);
			}

			fprintf(pfile, "\n");
		}
	}
}

int main(int argc, char ** argv)
{
	init_shogun_with_defaults();
	
	SGVector< float64_t > labs(NUM_CLASSES*NUM_SAMPLES);
	SGMatrix< float64_t > feats(DIMS, NUM_CLASSES*NUM_SAMPLES);

	gen_rand_data(labs, feats);

	// Create train labels
	CMulticlassSOLabels* labels = new CMulticlassSOLabels(labs);
	CMulticlassLabels*  mlabels = new CMulticlassLabels(labs);

	// Create train features
	CDenseFeatures< float64_t >* features = new CDenseFeatures< float64_t >(feats);

	// Create structured model
	CMulticlassModel* model = new CMulticlassModel(features, labels);
	model->set_use_bias(true);

	// Create loss function
	CHingeLoss* loss = new CHingeLoss();

	// Create SO-SVM
	CPrimalMosekSOSVM* sosvm = new CPrimalMosekSOSVM(model, loss, labels, features);
	SG_REF(sosvm);

	sosvm->train();
	CStructuredLabels* out = CStructuredLabels::obtain_from_generic(sosvm->apply());
	SG_REF(out);

	// Create liblinear svm classifier with L2-regularized L2-loss
	CLibLinear* svm = new CLibLinear(L2R_L2LOSS_SVC);

	// Add some configuration to the svm
	svm->set_epsilon(EPSILON);
	svm->set_bias_enabled(true);

	// Create a multiclass svm classifier that consists of several of the previous one
	CLinearMulticlassMachine* mc_svm = 
			new CLinearMulticlassMachine( new CMulticlassOneVsRestStrategy(), 
			(CDotFeatures*) features, svm, mlabels);
	SG_REF(mc_svm);

	// Train the multiclass machine using the data passed in the constructor
	mc_svm->train();
	CMulticlassLabels* mout = CMulticlassLabels::obtain_from_generic(mc_svm->apply());
	SG_REF(mout);

	int32_t sosvm_ncorrect = 0, mc_ncorrect = 0;
	SGVector< float64_t > slacks = sosvm->get_slacks();
	for ( int i = 0 ; i < out->get_num_labels() ; ++i )
	{
		SG_SPRINT("%.0f %.0f %.2E %.0f\n", mlabels->get_label(i), 
				( (CRealNumber*) out->get_label(i) )->value,
				slacks[i], mout->get_label(i));

		sosvm_ncorrect += mlabels->get_label(i) == ( (CRealNumber*) out->get_label(i) )->value;
		mc_ncorrect    += mlabels->get_label(i) == mout->get_label(i);
	}

	SG_SPRINT("\n");
	SGVector< float64_t > w = sosvm->get_w();
	for ( int32_t i = 0 ; i < w.vlen ; ++i )
		SG_SPRINT("%10f ", w[i]);
	SG_SPRINT("\n\n");

	for ( int32_t i = 0 ; i < NUM_CLASSES ; ++i )
	{
		SGVector< float64_t > mw = 
				((CLinearMachine*) mc_svm->get_machine(i))->get_w();
		for ( int32_t j = 0 ; j < mw.vlen ; ++j )
			SG_SPRINT("%10f ", mw[j]);
	}
	SG_SPRINT("\n");

	SG_SPRINT("SO-SVM: %5.2f%\n", 100.0*sosvm_ncorrect/mlabels->get_num_labels());
	SG_SPRINT("MC:     %5.2f%\n\n",  100.0*mc_ncorrect/mlabels->get_num_labels());

	// Free memory
	SG_UNREF(mout);
	SG_UNREF(mc_svm);
	SG_UNREF(sosvm);
	SG_UNREF(out);

	exit_shogun();

	return 0;
}
