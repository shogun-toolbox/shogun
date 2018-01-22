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

#include "MockObject.h"
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/regression/GaussianProcessRegression.h>
#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif
#include <gtest/gtest.h>

using namespace shogun;

TEST(SGObject, CCloneEqualsMock_allocate_delete)
{
	auto obj = some<CCloneEqualsMock<int32_t>>();
}

template <typename T>
class SGObjectEquals : public ::testing::Test
{
};
// TODO: SGString doesn't support complex128_t, so omitted here
typedef ::testing::Types<bool, char, int8_t, uint8_t, int16_t, int32_t, int64_t,
                         float32_t, float64_t, floatmax_t>
    CloneEqualsTypes;
TYPED_TEST_CASE(SGObjectEquals, CloneEqualsTypes);

TYPED_TEST(SGObjectEquals, same)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	EXPECT_TRUE(obj1->equals(obj1));
	EXPECT_TRUE(obj1->equals(obj2));
	EXPECT_TRUE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_null)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();

	EXPECT_FALSE(obj1->equals(nullptr));
}

TYPED_TEST(SGObjectEquals, different_basic)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_basic -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_object)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_object->m_some_value -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
	obj1->m_object->m_some_value += 1;

	delete obj1->m_object;
	obj1->m_object = nullptr;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_sg_vector)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_sg_vector[0] -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_sg_sparse_vector)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_sg_sparse_vector.features[0].entry -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_sg_sparse_matrix)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_sg_sparse_matrix.sparse_matrix[0].features[0].entry -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_sg_matrix)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_sg_matrix(0, 0) = 0;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_vector_basic)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_vector_basic[0] -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_vector_sg_string)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_vector_sg_string[0].string[0] -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_vector_object)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_vector_object[0]->m_some_value -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
	obj1->m_vector_object[0]->m_some_value += 1;

	delete obj1->m_vector_object[0];
	obj1->m_vector_object[0] = nullptr;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TEST(SGObject, equals_different_type)
{
	auto obj1 = some<CCloneEqualsMock<int>>();
	auto obj2 = some<CCloneEqualsMock<float>>();

	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TEST(SGObject, clone_not_null_and_correct_type)
{
	auto object = some<CGaussianKernel>();
	auto clone = object->clone();
	ASSERT_NE(clone, nullptr);
	ASSERT_TRUE(!strcmp(object->get_name(), clone->get_name()));
}

TEST(SGObject, clone_basic_parameter)
{
	auto object = some<CGaussianKernel>();
	object->put("log_width", 2.0);
	ASSERT_EQ(object->get<float64_t>("log_width"), 2);

	auto clone = object->clone();
	EXPECT_EQ(
	    object->get<float64_t>("log_width"),
	    clone->get<float64_t>("log_width"));
}

TEST(SGObject, clone_sgmatrix_parameter)
{
	SGMatrix<float64_t> data(10, 10);
	for (auto i : range(data.num_rows * data.num_cols))
		data.matrix[i] = CMath::randn_double();

	auto object = some<CDenseFeatures<float64_t>>(data);
	ASSERT_TRUE(data.equals(object->get_feature_matrix()));

	auto clone = object->clone();
	EXPECT_NE(
	    data.data(), clone->get<SGMatrix<float64_t>>("feature_matrix").data());
	ASSERT_TRUE(data.equals(clone->get<SGMatrix<float64_t>>("feature_matrix")));
}

TEST(SGObject, clone_sgvector_parameter)
{
	SGVector<float64_t> data(10);
	for (auto i : range(data.vlen))
		data.vector[i] = CMath::randn_double();

	auto object = some<CRegressionLabels>(data);
	ASSERT_TRUE(data.equals(object->get_labels()));

	auto clone = object->clone();
	EXPECT_NE(data.data(), clone->get<SGVector<float64_t>>("labels").data());
	EXPECT_TRUE(data.equals(clone->get<SGVector<float64_t>>("labels")));
}

TEST(SGObject, clone_sgobject_parameter)
{
	SGMatrix<float64_t> data(10, 10);
	for (auto i : range(data.num_rows * data.num_cols))
		data.matrix[i] = CMath::randn_double();

	auto object_parameter = some<CDenseFeatures<float64_t>>(data);
	auto object = some<CGaussianKernel>();
	object->init(object_parameter, object_parameter);
	ASSERT_TRUE(object_parameter->equals(object->get<CSGObject*>("lhs")));
	ASSERT_TRUE(object_parameter->equals(object->get<CSGObject*>("rhs")));

	auto clone = object->clone();
	EXPECT_NE(object_parameter, clone->get<CSGObject*>("lhs"));
	EXPECT_NE(object_parameter, clone->get<CSGObject*>("rhs"));
	EXPECT_TRUE(object_parameter->equals(clone->get<CSGObject*>("lhs")));
	EXPECT_TRUE(object_parameter->equals(clone->get<CSGObject*>("rhs")));
}

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
	set_global_fequals_epsilon(1e-10);
	EXPECT_TRUE(predictions->equals(predictions_copy));
	EXPECT_TRUE(gpr->equals(gpr_copy));
	set_global_fequals_epsilon(0);

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

	obj->put("vector", vec);
	EXPECT_THROW(obj->put("foo", vec), ShogunException);

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

	obj->put(Tag<SGVector<float64_t>>("vector"), vec);
	EXPECT_THROW(
	    obj->put(Tag<SGVector<float64_t>>("foo"), vec), ShogunException);
	EXPECT_THROW(obj->put(Tag<float64_t>("vector"), bar), ShogunException);

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
	obj->put("int", 10);
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

	obj->put("int", 10);
	EXPECT_EQ(obj->has(Tag<int32_t>("int")), true);
	EXPECT_EQ(obj->has(Tag<float64_t>("int")), false);
	EXPECT_EQ(obj->has("int"), true);
	EXPECT_EQ(obj->has<float64_t>("int"), false);
	EXPECT_EQ(obj->has<int32_t>("int"), true);

	EXPECT_EQ(obj->has("foo"), false);
	EXPECT_EQ(obj->has<int32_t>("foo"), false);
	EXPECT_EQ(obj->has(Tag<int32_t>("foo")), false);
}

TEST(SGObject, watched_parameter)
{
	auto obj = some<CMockObject>();

	obj->put("watched_int", 89);
	EXPECT_EQ(obj->get<int32_t>("watched_int"), 89);
	EXPECT_EQ(obj->get<int32_t>("watched_int"), obj->get_watched());
	obj->set_watched(12);
	EXPECT_EQ(obj->get<int32_t>("watched_int"), 12);
	EXPECT_EQ(obj->get<int32_t>("watched_int"), obj->get_watched());
}

TEST(SGObject, watched_parameter_object)
{
	auto obj = some<CMockObject>();
	Some<CMockObject> other_obj = some<CMockObject>();

	EXPECT_EQ(other_obj->ref_count(), 1);
	obj->put("watched_object", dynamic_cast<CSGObject*>(other_obj.get()));
	EXPECT_EQ(other_obj->ref_count(), 2);
	EXPECT_FALSE(other_obj->equals(obj));
	obj = nullptr;
	EXPECT_EQ(other_obj->ref_count(), 1);
}
