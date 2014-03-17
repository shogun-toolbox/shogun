/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
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
	CDenseLabels* labels=new CBinaryLabels(num_train+num_test);
	CMath::init_random(17);

	SGStringList<char> data(num_train+num_test, max_length);
	for (index_t i=0; i<num_train+num_test; ++i)
	{
		/* determine length */
		index_t length=CMath::random(1, max_length);

		/* allocate string */
		data.strings[i]=SGString<char>(length);

		/* fill with elements and set label */
		if (p_x<CMath::random(0.0, 1.0))
		{
			labels->set_label(i, 1);
			for (index_t j=0; j<length; ++j)
			{
				char c=mostly_prob<CMath::random(0.0, 1.0) ? '0' : '1';
				data.strings[i].string[j]=c;
			}
		}
		else
		{
			labels->set_label(i, -1);
			for (index_t j=0; j<length; ++j)
			{
				char c=mostly_prob<CMath::random(0.0, 1.0) ? '1' : '0';
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
	CLabels* labels_train=new CBinaryLabels(labels->get_labels_copy());
	labels->remove_subset();
	labels->add_subset(test_inds);
	CLabels* labels_test=new CBinaryLabels(labels->get_labels_copy());
	labels->remove_subset();

	/* string kernel */
	CDistantSegmentsKernel* kernel=new CDistantSegmentsKernel(10, 2, 2);

	/* SVM training and testing without precomputing the kernel */
	float64_t C=1;
	CSVM* svm=new CSVMLight(C, kernel, labels_train);
//	CSVM* svm=new CLibSVM(C, kernel, labels_train);
	svm->parallel->set_num_threads(1);
	svm->set_store_model_features(false);
	svm->train(feats_train);
	SGVector<float64_t> alphas=svm->get_alphas();
	SGVector<index_t> svs=svm->get_support_vectors();
	float64_t bias=svm->get_bias();
	CBinaryLabels* predictions=(CBinaryLabels*)svm->apply(feats_test);
	alphas.display_vector("alphas");
	svs.display_vector("svs");
	SG_SPRINT("bias: %f\n", bias);

	/* now the same with a precopumputed kernel */
	kernel->init(feats, feats);
	CCustomKernel* precomputed=new CCustomKernel(kernel);
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
	svm->set_kernel(new CCustomKernel(km_train));
	svm->train();
	SGVector<float64_t> alphas_precomputed=svm->get_alphas();
	SGVector<index_t> svs_precomputed=svm->get_support_vectors();
	float64_t bias_precomputed=svm->get_bias();
	alphas_precomputed.display_vector("alphas_precomputed");
	svs_precomputed.display_vector("svs_precomputed");
	SG_SPRINT("bias_precomputed: %f\n", bias_precomputed);
	svm->set_kernel(new CCustomKernel(km_test));
	CBinaryLabels* predictions_precomputed=(CBinaryLabels*)svm->apply();

	/* assert that the SV, alphas and b are equal, sort before (they may have
	 * a different ordering */
	CMath::qsort(alphas.vector, alphas.vlen);
	CMath::qsort(alphas_precomputed.vector, alphas_precomputed.vlen);
	CMath::qsort(svs.vector, svs.vlen);
	CMath::qsort(svs_precomputed.vector, svs_precomputed.vlen);

	ASSERT(alphas.vlen==alphas_precomputed.vlen);
	ASSERT(svs.vlen==svs_precomputed.vlen);
	for (index_t i=0; i<alphas.vlen; ++i)
	{
		ASSERT(CMath::abs(alphas[i]-alphas_precomputed[i])<1E-3);
		ASSERT(svs[i]==svs_precomputed[i]);
	}

	ASSERT(CMath::abs(bias-bias_precomputed)<1E-3);

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
	SG_UNREF(svm);
	SG_UNREF(precomputed);
	SG_UNREF(labels);
	SG_UNREF(labels_test);
	SG_UNREF(predictions);
	SG_UNREF(predictions_precomputed);
	SG_UNREF(feats_train);
	SG_UNREF(feats_test);
}

int main()
{
	init_shogun_with_defaults();
//	sg_io->set_loglevel(MSG_DEBUG);

	test_svmlight();

	exit_shogun();
	return 0;
}
#else
int main(int argc, char **argv)
{
	return 0;
}
#endif
