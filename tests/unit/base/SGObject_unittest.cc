/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 * Written (W) 2014 Thoralf Klein
 * Written (W) 2015 Wu Lin
 */

#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include "MockObject.h"
#include <shogun/base/some.h>
#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif
#include <gtest/gtest.h>

using namespace shogun;

TEST(SGObject,equals_same)
{
	Some<CGaussianKernel> kernel = some<CGaussianKernel>();
	EXPECT_TRUE(kernel->equals(kernel));
}

TEST(SGObject,equals_NULL_parameter)
{
	SGMatrix<float64_t> data(3,10);
	for (index_t i=0; i<data.num_rows*data.num_cols; ++i)
		data.matrix[i]=CMath::randn_double();

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);
	Some<CGaussianKernel> kernel = some<CGaussianKernel>();
	Some<CGaussianKernel> kernel2 = some<CGaussianKernel>();
	kernel2->init(feats, feats);

	EXPECT_FALSE(kernel->equals(kernel2));
}

TEST(SGObject,DISABLED_ref_copy_constructor)
{
	Some<CBinaryLabels> labs = some<CBinaryLabels>(10);

	// TODO: This causes memory corruptions; disabled test until fixed
	Some<CBinaryLabels> labs_2 = Some<CBinaryLabels>::from_raw(labs.get());

	CBinaryLabels* ptr = labs.get();
	EXPECT_TRUE(ptr == NULL);
}

TEST(SGObject,ref_unref_simple)
{
	CBinaryLabels* a = new CBinaryLabels(10);
	Some<CBinaryLabels> tmp = Some<CBinaryLabels>::from_raw(a);
	EXPECT_EQ(a->ref_count(), 1);
	if (tmp == NULL)
		EXPECT_TRUE(true);
}

TEST(SGObject,equals_null)
{
	Some<CBinaryLabels> labels = some<CBinaryLabels>(10);
	EXPECT_FALSE(labels->equals(NULL));
}

TEST(SGObject,equals_different_name)
{
	Some<CBinaryLabels> labels = some<CBinaryLabels>(10);
	Some<CRegressionLabels> labels2 = some<CRegressionLabels>(10);

	EXPECT_FALSE(labels->equals(labels2));
}

TEST(SGObject,equals_DynamicObjectArray_equal)
{
	Some<CDynamicObjectArray> array1 = some<CDynamicObjectArray>();
	Some<CDynamicObjectArray> array2 = some<CDynamicObjectArray>();

	CDynamicObjectArray* array1_tmp = array1.get();
	CDynamicObjectArray* array2_tmp = array2.get();

	EXPECT_TRUE(
	    TParameter::compare_ptype(PT_SGOBJECT, &array1_tmp, &array2_tmp));
}

TEST(SGObject,equals_DynamicObjectArray_equal_after_resize)
{
	Some<CDynamicObjectArray> array1 = some<CDynamicObjectArray>();
	Some<CDynamicObjectArray> array2 = some<CDynamicObjectArray>();

	CDynamicObjectArray* array1_tmp = array1.get();
	CDynamicObjectArray* array2_tmp = array2.get();

	/* enforce a resize */
	for (index_t i=0; i<1000; ++i)
		array1->append_element(new CGaussianKernel());

	array1->reset_array();

	EXPECT_TRUE(
	    TParameter::compare_ptype(PT_SGOBJECT, &array1_tmp, &array2_tmp));
}

TEST(SGObject,equals_DynamicObjectArray_different)
{
	Some<CDynamicObjectArray> array1 = some<CDynamicObjectArray>();
	Some<CDynamicObjectArray> array2 = some<CDynamicObjectArray>();

	CDynamicObjectArray* array1_tmp = array1.get();
	CDynamicObjectArray* array2_tmp = array2.get();

	array1->append_element(new CGaussianKernel());

	EXPECT_FALSE(
	    TParameter::compare_ptype(PT_SGOBJECT, &array1_tmp, &array2_tmp));
}

#ifdef USE_GPL_SHOGUN
TEST(SGObject,equals_complex_equal)
{
	/* create some easy regression data: 1d noisy sine wave */
	index_t n=100;
	float64_t x_range=6;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	for (index_t  i=0; i<n; ++i)
	{
		X[i]=CMath::random(0.0, x_range);
		X_test[i]=(float64_t)i / n*x_range;
		Y[i]=CMath::sin(X[i]);
	}

	/* shogun representation */
	Some<CDenseFeatures<float64_t>> feat_train =
	    some<CDenseFeatures<float64_t>>(X);
	Some<CDenseFeatures<float64_t>> feat_test =
	    some<CDenseFeatures<float64_t>>(X_test);
	Some<CRegressionLabels> label_train = some<CRegressionLabels>(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	Some<CGaussianKernel> kernel = some<CGaussianKernel>(10, shogun_sigma);
	Some<CZeroMean> mean = some<CZeroMean>();
	Some<CGaussianLikelihood> lik = some<CGaussianLikelihood>();
	lik->set_sigma(1);
	Some<CExactInferenceMethod> inf =
	    some<CExactInferenceMethod>(kernel, feat_train, mean, label_train, lik);
	Some<CGaussianProcessRegression> gpr =
	    some<CGaussianProcessRegression>(inf);

	/* train machine */
	gpr->train();

	// apply regression
	Some<CRegressionLabels> predictions =
	    Some<CRegressionLabels>::from_raw(gpr->apply_regression(feat_test));
	//predictions->get_labels().display_vector("predictions");

	/* save and load instance to compare */
	const char* filename_gpr="gpr_instance.txt";
	const char* filename_predictions="predictions_instance.txt";

	Some<CSerializableAsciiFile> file{nullptr};
	file=new CSerializableAsciiFile(filename_gpr, 'w');
	gpr->save_serializable(file);
	file->close();

	file=new CSerializableAsciiFile(filename_predictions, 'w');
	predictions->save_serializable(file);
	file->close();

	file=new CSerializableAsciiFile(filename_gpr, 'r');
	Some<CGaussianProcessRegression> gpr_copy =
	    some<CGaussianProcessRegression>();
	gpr_copy->load_serializable(file);
	file->close();

	file=new CSerializableAsciiFile(filename_predictions, 'r');
	Some<CRegressionLabels> predictions_copy = some<CRegressionLabels>();
	predictions_copy->load_serializable(file);
	file->close();

	/* now compare */
	floatmax_t accuracy=1E-10;
	EXPECT_TRUE(predictions->equals(predictions_copy, accuracy));
	EXPECT_TRUE(gpr->equals(gpr_copy, accuracy));

}
#endif //USE_GPL_SHOGUN

#ifdef USE_GPL_SHOGUN
TEST(SGObject,update_parameter_hash)
{
	index_t n=3;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	X[0]=0;
	X[1]=1.1;
	X[2]=2.2;

	X_test[0]=0.3;
	X_test[1]=1.3;
	X_test[2]=2.5;

	for (index_t i=0; i<n; ++i)
	{
		Y[i]=CMath::sin(X(0, i));
	}

	Some<CDenseFeatures<float64_t>> feat_train =
	    some<CDenseFeatures<float64_t>>(X);
	Some<CRegressionLabels> label_train = some<CRegressionLabels>(Y);

	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	Some<CGaussianKernel> kernel = some<CGaussianKernel>(10, shogun_sigma);
	Some<CZeroMean> mean = some<CZeroMean>();
	Some<CGaussianLikelihood> lik = some<CGaussianLikelihood>();
	lik->set_sigma(1);
	Some<CExactInferenceMethod> inf =
	    some<CExactInferenceMethod>(kernel, feat_train, mean, label_train, lik);

	SGMatrix<float64_t> L=inf->get_cholesky();
	uint32_t hash1=inf->m_hash;

	inf->update_parameter_hash();
	inf->update_parameter_hash();
	uint32_t hash2=inf->m_hash;

	EXPECT_TRUE(hash1==hash2);

}
#endif //USE_GPL_SHOGUN

#ifdef USE_GPL_SHOGUN
TEST(SGObject,parameter_hash_changed)
{
	index_t n=3;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	X[0]=0;
	X[1]=1.1;
	X[2]=2.2;

	X_test[0]=0.3;
	X_test[1]=1.3;
	X_test[2]=2.5;

	for (index_t i=0; i<n; ++i)
	{
		Y[i]=CMath::sin(X(0, i));
	}

	Some<CDenseFeatures<float64_t>> feat_train =
	    some<CDenseFeatures<float64_t>>(X);
	Some<CRegressionLabels> label_train = some<CRegressionLabels>(Y);

	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	Some<CGaussianKernel> kernel = some<CGaussianKernel>(10, shogun_sigma);
	Some<CZeroMean> mean = some<CZeroMean>();
	Some<CGaussianLikelihood> lik = some<CGaussianLikelihood>();
	lik->set_sigma(1);
	Some<CExactInferenceMethod> inf =
	    some<CExactInferenceMethod>(kernel, feat_train, mean, label_train, lik);
	EXPECT_TRUE(inf->parameter_hash_changed());

	SGMatrix<float64_t> L=inf->get_cholesky();
	EXPECT_FALSE(inf->parameter_hash_changed());

	inf->update_parameter_hash();
	inf->update_parameter_hash();
	EXPECT_FALSE(inf->parameter_hash_changed());

}
#endif //USE_GPL_SHOGUN

TEST(SGObject, tags_set_get_string_sgvector)
{
	auto obj = some<CMockObject>();
	auto vec = SGVector<float64_t>(1);
	vec[0] = 1;

	obj->set("vector", vec);
	EXPECT_THROW(obj->set("foo", vec), ShogunException);

	auto retr = obj->get<SGVector<float64_t> >("vector");

	EXPECT_EQ(retr.vlen, vec.vlen);
	EXPECT_EQ(vec[0], retr[0]);
	EXPECT_THROW(obj->get(Tag<SGVector<int32_t> >("vector")), ShogunException);
	EXPECT_THROW(obj->get<SGVector<int32_t> >("vector"), ShogunException);
}

TEST(SGObject, tags_set_get_tag_sgvector)
{
	auto obj = some<CMockObject>();
	auto vec = SGVector<float64_t>(1);
	vec[0] = 1;
	float64_t bar = 1.0;

	obj->set(Tag<SGVector<float64_t> >("vector"), vec);
	EXPECT_THROW(obj->set(Tag<SGVector<float64_t> >("foo"), vec), ShogunException);
	EXPECT_THROW(obj->set(Tag<float64_t>("vector"), bar), ShogunException);

	auto retr = obj->get<SGVector<float64_t> >("vector");

	EXPECT_EQ(retr.vlen, vec.vlen);
	EXPECT_EQ(vec[0], retr[0]);
	EXPECT_THROW(obj->get(Tag<SGVector<int32_t> >("vector")), ShogunException);
	EXPECT_THROW(obj->get<SGVector<int32_t> >("vector"), ShogunException);
}

TEST(SGObject, tags_set_get_int)
{
	auto obj = some<CMockObject>();

	EXPECT_THROW(obj->get<int32_t>("foo"), ShogunException);
	obj->set("int", 10);
	EXPECT_EQ(obj->get(Tag<int32_t>("int")), 10);
	EXPECT_THROW(obj->get<float64_t>("int"), ShogunException);
	EXPECT_THROW(obj->get(Tag<float64_t>("int")), ShogunException);
	EXPECT_EQ(obj->get<int>("int"), 10);
}

TEST(SGObject, tags_has)
{
	auto obj = some<CMockObject>();

	EXPECT_EQ(obj->has(Tag<int32_t>("int")), true);
	EXPECT_EQ(obj->has(Tag<float64_t>("int")), false);
	EXPECT_EQ(obj->has("int"), true);
	EXPECT_EQ(obj->has<float64_t>("int"), false);
	EXPECT_EQ(obj->has<int32_t>("int"), true);

	obj->set("int", 10);
	EXPECT_EQ(obj->has(Tag<int32_t>("int")), true);
	EXPECT_EQ(obj->has(Tag<float64_t>("int")), false);
	EXPECT_EQ(obj->has("int"), true);
	EXPECT_EQ(obj->has<float64_t>("int"), false);
	EXPECT_EQ(obj->has<int32_t>("int"), true);

	EXPECT_EQ(obj->has("foo"), false);
	EXPECT_EQ(obj->has<int32_t>("foo"), false);
	EXPECT_EQ(obj->has(Tag<int32_t>("foo")), false);
}
