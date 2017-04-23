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
	CGaussianKernel* kernel=new CGaussianKernel();
	EXPECT_TRUE(kernel->equals(kernel));
	SG_UNREF(kernel);
}

TEST(SGObject,equals_NULL_parameter)
{
	SGMatrix<float64_t> data(3,10);
	for (index_t i=0; i<data.num_rows*data.num_cols; ++i)
		data.matrix[i]=CMath::randn_double();

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);
	CGaussianKernel* kernel=new CGaussianKernel();
	CGaussianKernel* kernel2=new CGaussianKernel();
	kernel2->init(feats, feats);

	EXPECT_FALSE(kernel->equals(kernel2));

	SG_UNREF(kernel);
	SG_UNREF(kernel2);
}

#ifdef USE_REFERENCE_COUNTING
TEST(SGObject,DISABLED_ref_copy_constructor)
{
	CBinaryLabels* labs = new CBinaryLabels(10);
	EXPECT_EQ(labs->ref_count(), 0);

	SG_REF(labs);
	EXPECT_EQ(labs->ref_count(), 1);

	// TODO: This causes memory corruptions; disabled test until fixed
	CBinaryLabels* labs_2 = new CBinaryLabels(*labs);
	SG_UNREF(labs_2);

	SG_UNREF(labs);
	EXPECT_TRUE(labs == NULL);
}

TEST(SGObject,ref_unref_simple)
{
	CBinaryLabels* labs = new CBinaryLabels(10);
	EXPECT_EQ(labs->ref_count(), 0);

	SG_REF(labs);
	EXPECT_EQ(labs->ref_count(), 1);

	SG_UNREF(labs);
	EXPECT_TRUE(labs == NULL);
}
#endif

TEST(SGObject,equals_null)
{
	CBinaryLabels* labels=new CBinaryLabels(10);

	EXPECT_FALSE(labels->equals(NULL));

	SG_UNREF(labels);
}

TEST(SGObject,equals_different_name)
{
	CBinaryLabels* labels=new CBinaryLabels(10);
	CRegressionLabels* labels2=new CRegressionLabels(10);

	EXPECT_FALSE(labels->equals(labels2));

	SG_UNREF(labels);
	SG_UNREF(labels2);
}

TEST(SGObject,equals_DynamicObjectArray_equal)
{
	CDynamicObjectArray* array1=new CDynamicObjectArray();
	CDynamicObjectArray* array2=new CDynamicObjectArray();

	EXPECT_TRUE(TParameter::compare_ptype(PT_SGOBJECT, &array1, &array2));

	SG_UNREF(array1);
	SG_UNREF(array2);
}

TEST(SGObject,equals_DynamicObjectArray_equal_after_resize)
{
	CDynamicObjectArray* array1=new CDynamicObjectArray();
	CDynamicObjectArray* array2=new CDynamicObjectArray();

	/* enforce a resize */
	for (index_t i=0; i<1000; ++i)
		array1->append_element(new CGaussianKernel());

	array1->reset_array();

	EXPECT_TRUE(TParameter::compare_ptype(PT_SGOBJECT, &array1, &array2));

	SG_UNREF(array1);
	SG_UNREF(array2);
}

TEST(SGObject,equals_DynamicObjectArray_different)
{
	CDynamicObjectArray* array1=new CDynamicObjectArray();
	CDynamicObjectArray* array2=new CDynamicObjectArray();

	array1->append_element(new CGaussianKernel());

	EXPECT_FALSE(TParameter::compare_ptype(PT_SGOBJECT, &array1, &array2));

	SG_UNREF(array1);
	SG_UNREF(array2);
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
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	lik->set_sigma(1);
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);
	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

	/* train machine */
	gpr->train();

	// apply regression
	CRegressionLabels* predictions=gpr->apply_regression(feat_test);
	//predictions->get_labels().display_vector("predictions");

	/* save and load instance to compare */
	const char* filename_gpr="gpr_instance.txt";
	const char* filename_predictions="predictions_instance.txt";

	CSerializableAsciiFile* file;
	file=new CSerializableAsciiFile(filename_gpr, 'w');
	gpr->save_serializable(file);
	file->close();
	SG_UNREF(file);

	file=new CSerializableAsciiFile(filename_predictions, 'w');
	predictions->save_serializable(file);
	file->close();
	SG_UNREF(file);

	file=new CSerializableAsciiFile(filename_gpr, 'r');
	CGaussianProcessRegression* gpr_copy=new CGaussianProcessRegression();
	gpr_copy->load_serializable(file);
	file->close();
	SG_UNREF(file);

	file=new CSerializableAsciiFile(filename_predictions, 'r');
	CRegressionLabels* predictions_copy=new CRegressionLabels();
	predictions_copy->load_serializable(file);
	file->close();
	SG_UNREF(file);

	/* now compare */
	floatmax_t accuracy=1E-10;
	EXPECT_TRUE(predictions->equals(predictions_copy, accuracy));
	EXPECT_TRUE(gpr->equals(gpr_copy, accuracy));

	SG_UNREF(predictions);
	SG_UNREF(predictions_copy);
	SG_UNREF(gpr);
	SG_UNREF(gpr_copy);
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

	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	lik->set_sigma(1);
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);

	SGMatrix<float64_t> L=inf->get_cholesky();
	uint32_t hash1=inf->m_hash;

	inf->update_parameter_hash();
	inf->update_parameter_hash();
	uint32_t hash2=inf->m_hash;

	EXPECT_TRUE(hash1==hash2);

	SG_UNREF(inf);
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

	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	lik->set_sigma(1);
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);
	EXPECT_TRUE(inf->parameter_hash_changed());

	SGMatrix<float64_t> L=inf->get_cholesky();
	EXPECT_FALSE(inf->parameter_hash_changed());

	inf->update_parameter_hash();
	inf->update_parameter_hash();
	EXPECT_FALSE(inf->parameter_hash_changed());

	SG_UNREF(inf);
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
