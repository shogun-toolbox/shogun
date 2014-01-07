#include <base/init.h>
#include <features/RandomFourierDotFeatures.h>
#include <kernel/GaussianKernel.h>
#include <kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <classifier/svm/LibLinear.h>
#include <classifier/svm/SVMOcas.h>
#include <classifier/svm/LibSVM.h>
#include <labels/BinaryLabels.h>
#include <evaluation/PRCEvaluation.h>
#include <lib/Time.h>

#include <stdio.h>
#include <ctime>

using namespace shogun;

/** Code that compares the times needed to train
 * a linear svm using the RandomFourierDotFeatures class
 * vs a non-linear svm using the Gaussian Kernel.
 */
int main(int argv, char** argc)
{
	init_shogun_with_defaults();

	int32_t dims[] = {10, 100, 1000};
	int32_t vecs[] = {10000, 100000, 1000000};
	CTime* timer = new CTime(false);
	float64_t epsilon = 0.001;
	float64_t lin_C = 0.1;
	float64_t non_lin_C = 0.1;
	CPRCEvaluation* evaluator = new CPRCEvaluation();
	CSqrtDiagKernelNormalizer* normalizer = new CSqrtDiagKernelNormalizer(true);
	SG_REF(normalizer);
	for (index_t d=0; d<4; d++)
	{
		int32_t num_dim = dims[d];
		SG_SPRINT("Starting experiment for number of dimensions = %d\n", num_dim);
		for (index_t v=0; v<3; v++)
		{
			int32_t num_vecs = vecs[v];
			SG_SPRINT("   Using %d examples\n", num_vecs);
			SGMatrix<float64_t> mat(num_dim, num_vecs);
			SGVector<float64_t> labs(num_vecs);
			for (index_t i=0; i<num_vecs; i++)
			{
				for (index_t j=0; j<num_dim; j++)
				{
					if ((i+j)%2==0)
					{
						labs[i] = -1;
						mat(j,i) = CMath::random(0,1) + 0.5;
					}
					else
					{
						labs[i] = 1;
						mat(j,i) = CMath::random(0,1) - 0.5;
					}
				}
			}

			SGVector<float64_t> params(1);
			params[0] = 8;
			SG_SPRINT("    Using kernel_width = %f\n", params[0]);

			CDenseFeatures<float64_t>* dense_feats = new CDenseFeatures<float64_t>(mat);
			SG_REF(dense_feats);

			CBinaryLabels* labels = new CBinaryLabels(labs);
			SG_REF(labels);

			/** LibLinear SVM using RandomFourierDotFeatures */
			int32_t D[] = {50, 100, 300, 1000};
			for (index_t d=0; d<4; d++)
			{
				CRandomFourierDotFeatures* r_feats = new CRandomFourierDotFeatures(
						dense_feats, D[d], KernelName::GAUSSIAN, params);

				//CLibLinear* lin_svm = new CLibLinear(C, r_feats, labels);
				CSVMOcas* lin_svm = new CSVMOcas(lin_C, r_feats, labels);
				lin_svm->set_epsilon(epsilon);
				clock_t t = clock();
				timer->start();
				lin_svm->train();
				t = clock() - t;
				timer->stop();
				SG_SPRINT("\tSVMOcas using RFDotFeatures(D=%d) finished training. Took %fs (or %fs), ",
						D[d], timer->time_diff_sec(), (float64_t) t /CLOCKS_PER_SEC);

				t = clock();
				timer->start();
				CBinaryLabels* predicted = CLabelsFactory::to_binary(lin_svm->apply());
				timer->stop();
				t = clock() - t;
				float64_t auPRC = evaluator->evaluate(predicted, labels);
				SG_SPRINT("SVMOcas auPRC=%f (Applying took %fs (%fs)\n", auPRC,
						timer->time_diff_sec(), (float64_t) t / CLOCKS_PER_SEC);
				SG_UNREF(lin_svm);
				SG_UNREF(predicted);
			}
			/** End of LibLinear code */


			/** LibSVM using Gaussian Kernel */

			CGaussianKernel* kernel = new CGaussianKernel(dense_feats, dense_feats, params[0]);
			//kernel->set_normalizer(normalizer);
			CLibSVM* svm = new CLibSVM(non_lin_C, kernel, labels);
			svm->set_epsilon(epsilon);
			clock_t t = clock();
			timer->start();
			svm->train();
			t = clock() - t;
			timer->stop();
			SG_SPRINT("\tLibSVM using GaussianKernel finished training. Took %fs (or %fs), ",
					timer->time_diff_sec(), (float64_t) t /CLOCKS_PER_SEC);

			t = clock();
			timer->start();
			CBinaryLabels* predicted = CLabelsFactory::to_binary(svm->apply());
			timer->stop();
			t = clock() - t;
			float64_t auPRC = evaluator->evaluate(predicted, labels);
			SG_SPRINT("LibSVM auPRC=%f (Applying took %fs (%fs)\n", auPRC,
					timer->time_diff_sec(), (float64_t) t / CLOCKS_PER_SEC);
			SG_UNREF(svm);
			SG_UNREF(predicted);
			/** End of LibSVM code */
			SG_UNREF(labels);
			SG_UNREF(dense_feats);
		}
	}
	SG_UNREF(timer);
	SG_UNREF(evaluator);
	SG_UNREF(normalizer);
	exit_shogun();
}
