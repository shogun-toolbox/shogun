/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Thoralf Klein, Heiko Strathmann
 */

#include <shogun/lib/config.h>

#include <shogun/kernel/string/DistantSegmentsKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/classifier/svm/SVMLight.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/lib/SGStringList.h>

using namespace shogun;

#ifdef USE_SVMLIGHT
void test_svmlight()
{
	/* data is random length strings with only zeros (A) or ones (B) */
	index_t num_train=100;
	index_t num_test=50;
	index_t max_length=100;
	float64_t p_x=0.5; // probability for class A
	float64_t mostly_prob=0.8;
	DenseLabels* labels=new BinaryLabels(num_train+num_test);
	Math::init_random(17);

	SGStringList<char> data(num_train+num_test, max_length);
	for (index_t i=0; i<num_train+num_test; ++i)
	{
		/* determine length */
		index_t length=Math::random(1, max_length);

		/* allocate string */
		data.strings[i]=SGString<char>(length);

		/* fill with elements and set label */
		if (p_x<Math::random(0.0, 1.0))
		{
			labels->set_label(i, 1);
			for (index_t j=0; j<length; ++j)
			{
				char c=mostly_prob<Math::random(0.0, 1.0) ? '0' : '1';
				data.strings[i].string[j]=c;
			}
		}
		else
		{
			labels->set_label(i, -1);
			for (index_t j=0; j<length; ++j)
			{
				char c=mostly_prob<Math::random(0.0, 1.0) ? '1' : '0';
				data.strings[i].string[j]=c;
			}
		}

		SG_SPRINT("datum %d, class %d:\t", i, labels->get_int_label(i));
		for (index_t j=0; j<length; ++j)
			SG_SPRINT("%c", data.strings[i].string[j]);
		SG_SPRINT("\n");

	}
	CStringFeatures<char>* feats=new CStringFeatures<char>(data, BINARY);

	/* copy training and test data */
	SGVector<index_t> train_inds(num_train);
	train_inds.range_fill();
	SGVector<index_t> test_inds(num_test);
	test_inds.range_fill();
	test_inds.add(num_train);

	CStringFeatures<char>* feats_train=
			(CStringFeatures<char>*)feats->copy_subset(train_inds);
	CStringFeatures<char>* feats_test=
			(CStringFeatures<char>*)feats->copy_subset(test_inds);

	labels->add_subset(train_inds);
	Labels* labels_train=new BinaryLabels(labels->get_labels_copy());
	labels->remove_subset();
	labels->add_subset(test_inds);
	Labels* labels_test=new BinaryLabels(labels->get_labels_copy());
	labels->remove_subset();

	/* string kernel */
	CDistantSegmentsKernel* kernel=new CDistantSegmentsKernel(10, 2, 2);

	/* SVM training and testing without precomputing the kernel */
	float64_t C=1;
	SVM* svm=new SVMLight(C, kernel, labels_train);
	svm->get_global_parallel()->set_num_threads(1);
	svm->set_store_model_features(false);
	svm->train(feats_train);
	SGVector<float64_t> alphas=svm->get_alphas();
	SGVector<index_t> svs=svm->get_support_vectors();
	float64_t bias=svm->get_bias();
	BinaryLabels* predictions=(BinaryLabels*)svm->apply(feats_test);
	alphas.display_vector("alphas");
	svs.display_vector("svs");
	SG_SPRINT("bias: %f\n", bias);

	/* now the same with a precopumputed kernel */
	kernel->init(feats, feats);
	CustomKernel* precomputed=new CustomKernel(kernel);
	precomputed->add_row_subset(train_inds);
	precomputed->add_col_subset(train_inds);
	SGMatrix<float64_t> km_train=precomputed->get_kernel_matrix();
	precomputed->remove_col_subset();
	precomputed->add_col_subset(test_inds);
	SGMatrix<float64_t> km_test=precomputed->get_kernel_matrix();
	precomputed->remove_row_subset();
	precomputed->remove_col_subset();
	SGMatrix<float64_t> km=precomputed->get_kernel_matrix();

//	km.display_matrix("FULL");
//	km_train.display_matrix("TRAIN");
//	km_test.display_matrix("TEST");

	/* make sure matrices are correct */
	for (index_t i=0; i<km_train.num_rows; ++i)
	{
		for (index_t j=0; j<km_train.num_cols; ++j)
			ASSERT(km_train(i, j)==km(i, j));
	}

	for (index_t i=0; i<km_test.num_rows; ++i)
	{
		for (index_t j=0; j<km_test.num_cols; ++j)
			ASSERT(km_test(i, j)==km(i, j+num_train));
	}

	/* train and test again on custom kernel */
	svm->set_kernel(new CustomKernel(km_train));
	svm->train();
	SGVector<float64_t> alphas_precomputed=svm->get_alphas();
	SGVector<index_t> svs_precomputed=svm->get_support_vectors();
	float64_t bias_precomputed=svm->get_bias();
	alphas_precomputed.display_vector("alphas_precomputed");
	svs_precomputed.display_vector("svs_precomputed");
	SG_SPRINT("bias_precomputed: %f\n", bias_precomputed);
	svm->set_kernel(new CustomKernel(km_test));
	BinaryLabels* predictions_precomputed=(BinaryLabels*)svm->apply();

	/* assert that the SV, alphas and b are equal, sort before (they may have
	 * a different ordering */
	Math::qsort(alphas.vector, alphas.vlen);
	Math::qsort(alphas_precomputed.vector, alphas_precomputed.vlen);
	Math::qsort(svs.vector, svs.vlen);
	Math::qsort(svs_precomputed.vector, svs_precomputed.vlen);

	ASSERT(alphas.vlen==alphas_precomputed.vlen);
	ASSERT(svs.vlen==svs_precomputed.vlen);
	for (index_t i=0; i<alphas.vlen; ++i)
	{
		ASSERT(Math::abs(alphas[i]-alphas_precomputed[i])<1E-3);
		ASSERT(svs[i]==svs_precomputed[i]);
	}

	ASSERT(Math::abs(bias-bias_precomputed)<1E-3);

	/* assert that predictions are the same */
	predictions->get_int_labels().display_vector("predictions");
	predictions_precomputed->get_int_labels().
			display_vector("predictions_precomputed");

	for (index_t i=0; i<predictions->get_num_labels(); ++i)
	{
		ASSERT(predictions->get_int_label(i)==
				predictions_precomputed->get_int_label(i));
	}

	/* clean up */
	SG_SPRINT("cleaning up\n");
}

int main()
{
//	env()->io()->set_loglevel(MSG_DEBUG);

	test_svmlight();

	return 0;
}
#else
int main(int argc, char **argv)
{
	return 0;
}
#endif
