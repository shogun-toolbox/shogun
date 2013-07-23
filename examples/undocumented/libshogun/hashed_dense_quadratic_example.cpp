#include <shogun/base/init.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/HashedDenseFeatures.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/io/AsciiFile.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/evaluation/PRCEvaluation.h>
#include <shogun/evaluation/ROCEvaluation.h>

using namespace shogun;

void gen_rand_data(SGVector<float64_t> lab, SGMatrix<float64_t> feat,
		float64_t dist)
{
	index_t dims=feat.num_rows;
	index_t num=lab.vlen;

	for (int32_t i=0; i<num; i++)
	{
		if (i<num/2)
		{
			lab[i]=-1.0;

			for (int32_t j=0; j<dims; j++)
				feat(j, i)=CMath::random(0.0, 1.0)+dist;
		}
		else
		{
			lab[i]=1.0;

			for (int32_t j=0; j<dims; j++)
				feat(j, i)=CMath::random(0.0, 1.0)-dist;
		}
	}
}

void train_poly_kernel_svm(CDenseFeatures<float64_t>* dense_feats, CLabels* labels)
{
	SG_SPRINT("Starting CPolyKernel svm test\n");
	clock_t t = clock();

	SG_REF(dense_feats);
	SG_REF(labels);
	CPolyKernel* poly_kernel = new CPolyKernel(dense_feats, dense_feats, 2, true, 10);
	float64_t c = 10;
	float64_t epsilon = 0.001;
	CLibSVM* lib_svm = new CLibSVM(c, poly_kernel, labels); 
	lib_svm->set_epsilon(epsilon);

	SG_SPRINT("Start training..\n");
	lib_svm->train();
	SG_SPRINT("Training completed\nEvaluating..\n");

	CBinaryLabels* out_labels=CLabelsFactory::to_binary(lib_svm->apply());

	t = clock() - t;

	CPRCEvaluation* prc_evaluator = new CPRCEvaluation();
	float64_t auPRC = prc_evaluator->evaluate(out_labels, labels);
	SG_UNREF(prc_evaluator);

	CROCEvaluation* roc_evaluator = new CROCEvaluation();
	float64_t auROC = roc_evaluator->evaluate(out_labels, labels);
	SG_UNREF(out_labels);
	SG_UNREF(roc_evaluator);	

	SG_SPRINT("Results for poly kernel :\n");
	SG_SPRINT("auPRC = %f, auROC = %f\n", auPRC, auROC);
	SG_SPRINT("It took %d clicks or %f seconds\n\n",t,((float)t)/CLOCKS_PER_SEC);

	SG_UNREF(lib_svm);
}

void train_hashed_dense_feats(CDenseFeatures<float64_t>* dense_feats, CLabels* labels)
{
	SG_SPRINT("Starting CHashedDenseFeatures test\n");
	clock_t t = clock();

	SG_REF(dense_feats);
	SG_REF(labels);
	int32_t hashing_dim = 1000;
	CHashedDenseFeatures<float64_t>* feats = new CHashedDenseFeatures<float64_t>(dense_feats,
		hashing_dim, true, false);
	/*SG_SPRINT("be1\n");
	feats->benchmark_add_to_dense_vector();
	SG_SPRINT("be2\n");
	feats->benchmark_dense_dot_range();*/
	float64_t c = 10;
	float64_t epsilon = 0.1;
	CSVMOcas* svm_ocas = new CSVMOcas(c, feats, labels); 
	svm_ocas->set_epsilon(epsilon);
	SG_SPRINT("Start training..\n");
	svm_ocas->train();
	SG_SPRINT("Training completed\nEvaluating..\n");

	CBinaryLabels* out_labels=CLabelsFactory::to_binary(svm_ocas->apply());

	t = clock() - t;

	CPRCEvaluation* prc_evaluator = new CPRCEvaluation();
	float64_t auPRC = prc_evaluator->evaluate(out_labels, labels);
	SG_UNREF(prc_evaluator);

	CROCEvaluation* roc_evaluator = new CROCEvaluation();
	float64_t auROC = roc_evaluator->evaluate(out_labels, labels);
	SG_UNREF(out_labels);
	SG_UNREF(roc_evaluator);	

	SG_SPRINT("Results for HashedDenseFeatures :\n");
	SG_SPRINT("auPRC = %f, auROC = %f\n", auPRC, auROC);
	SG_SPRINT("It took %d clicks or %f seconds\n\n",t,((float)t)/CLOCKS_PER_SEC);

	SG_UNREF(svm_ocas);
}
int main()
{
	init_shogun_with_defaults();

	int num_vectors = 5000;
	int dimension = 2000;

	SGMatrix<float64_t> data(dimension, num_vectors);
	SGVector<float64_t> lab(num_vectors);

	gen_rand_data(lab, data, 0.5);
	CDenseFeatures<float64_t>* dense_feats = new CDenseFeatures<float64_t>(data);
	CLabels* labels=new CBinaryLabels(lab);

//	train_poly_kernel_svm(dense_feats, labels);
	SG_SPRINT("---------------------------------\n");
	train_hashed_dense_feats(dense_feats, labels);

	SG_UNREF(dense_feats);
	SG_UNREF(labels);
	exit_shogun();
	return 0;
}
