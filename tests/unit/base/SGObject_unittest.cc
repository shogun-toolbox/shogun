/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Thoralf Klein, Wu Lin
 */

#include "sg_gtest_utilities.h"

#include "MockObject.h"
#include <shogun/base/class_list.h>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/observers/ParameterObserverScalar.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/regression/GaussianProcessRegression.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif


using namespace shogun;

// fixture for SGObject::equals tests
template <typename T>
class SGObjectEquals : public ::testing::Test
{
};

// fixture for SGObject::clone tests
template <typename T>
class SGObjectClone : public ::testing::Test
{
};

SG_TYPED_TEST_CASE(SGObjectEquals, sg_all_primitive_types, complex128_t);
SG_TYPED_TEST_CASE(SGObjectClone, sg_all_primitive_types, complex128_t);

TYPED_TEST(SGObjectEquals, mock_allocate_delete)
{
	auto obj = some<CCloneEqualsMock<TypeParam>>();
}

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

TYPED_TEST(SGObjectEquals, different_string)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_string = "Oh no!";
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

TYPED_TEST(SGObjectEquals, different_raw_vector_basic)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_raw_vector_basic[0] -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_raw_vector_sg_string)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_raw_vector_sg_string[0].string[0] -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_raw_vector_object)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_raw_vector_object[0]->m_some_value -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
	obj1->m_raw_vector_object[0]->m_some_value += 1;

	delete obj1->m_raw_vector_object[0];
	obj1->m_raw_vector_object[0] = nullptr;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_raw_matrix_basic)
{
	auto obj1 = some<CCloneEqualsMock<TypeParam>>();
	auto obj2 = some<CCloneEqualsMock<TypeParam>>();

	obj1->m_raw_matrix_basic[0] -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TEST(SGObjectEquals, different_type)
{
	auto obj1 = some<CCloneEqualsMock<int>>();
	auto obj2 = some<CCloneEqualsMock<float>>();

	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectClone, basic_checks)
{
	auto obj = some<CCloneEqualsMock<TypeParam>>();

	CSGObject* clone = obj->clone();

	EXPECT_NE(clone, obj);
	ASSERT_NE(clone, nullptr);
	EXPECT_EQ(clone->ref_count(), 1);

	auto clone_casted = clone->as<CCloneEqualsMock<TypeParam>>();
	ASSERT_NE(clone_casted, nullptr);

	EXPECT_EQ(std::string(clone->get_name()), std::string(obj->get_name()));

	SG_UNREF(clone);
}

TYPED_TEST(SGObjectClone, equals_empty)
{
	auto obj = some<CCloneEqualsMock<TypeParam>>();

	CSGObject* clone = obj->clone();
	EXPECT_TRUE(clone->equals(obj));

	SG_UNREF(clone);
}

TYPED_TEST(SGObjectClone, equals_non_empty)
{
	auto obj = some<CCloneEqualsMock<TypeParam>>();
	obj->m_basic -= 1;
	obj->m_string = "Non empty string";
	obj->m_object->m_some_value -= 1;
	obj->m_sg_vector[0] -= 1;
	obj->m_sg_matrix(0, 0) = 0;
	obj->m_sg_sparse_vector.features[0].entry -= 1;
	obj->m_sg_sparse_matrix.sparse_matrix[0].features[0].entry -= 1;
	obj->m_raw_vector_basic[0] -= 1;
	obj->m_raw_matrix_basic[0] -= 1;
	obj->m_raw_vector_sg_string[0].string[0] -= 1;
	obj->m_raw_vector_object[0]->m_some_value -= 1;

	CSGObject* clone = obj->clone();
	EXPECT_TRUE(clone->equals(obj));

	SG_UNREF(clone);
}

TYPED_TEST(SGObjectClone, not_just_copied_pointer)
{
	auto obj = some<CCloneEqualsMock<TypeParam>>();
	CSGObject* clone = obj->clone();
	auto clone_casted = clone->as<CCloneEqualsMock<TypeParam>>();
	ASSERT_NE(clone_casted, nullptr);

	EXPECT_NE(clone_casted->m_object, obj->m_object);
	EXPECT_NE(clone_casted->m_raw_vector_basic, obj->m_raw_vector_basic);
	EXPECT_NE(clone_casted->m_raw_vector_sg_string, obj->m_raw_vector_sg_string);

	EXPECT_NE(clone_casted->m_raw_vector_object, obj->m_raw_vector_object);
	for (auto i : range(obj->m_raw_vector_object_len))
		EXPECT_NE(clone_casted->m_raw_vector_object[i], obj->m_raw_vector_object[i]);

	SG_UNREF(clone);
}

TYPED_TEST(SGObjectClone, equals_other_has_null_param)
{
	auto obj = some<CCloneEqualsMock<TypeParam>>();
	SG_UNREF(obj->m_object);

	CSGObject* clone = obj->clone();
	EXPECT_TRUE(clone->equals(obj));

	SG_UNREF(clone);
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
		Y[i] = std::sin(X[i]);
	}

	/* shogun representation */
	auto feat_train = some<CDenseFeatures<float64_t>>(X);
	auto feat_test = some<CDenseFeatures<float64_t>>(X_test);
	auto label_train = some<CRegressionLabels>(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	auto kernel = some<CGaussianKernel>(10, shogun_sigma);
	auto mean = some<CZeroMean>();
	auto lik = some<CGaussianLikelihood>();
	lik->set_sigma(1);
	auto inf = some<CExactInferenceMethod>(kernel, feat_train,
			mean, label_train, lik);
	auto gpr = some<CGaussianProcessRegression>(inf);

	/* train machine */
	gpr->train();

	// apply regression
	CRegressionLabels* predictions=gpr->apply_regression(feat_test);
	//predictions->get_labels().display_vector("predictions");

	/* save and load instance to compare */
	const char* filename_gpr="gpr_instance.txt";
	const char* filename_predictions="predictions_instance.txt";

	auto file = some<CSerializableAsciiFile>(filename_gpr, 'w');
	gpr->save_serializable(file);
	file->close();

	file = some<CSerializableAsciiFile>(filename_predictions, 'w');
	predictions->save_serializable(file);
	file->close();

	file = some<CSerializableAsciiFile>(filename_gpr, 'r');
	auto gpr_copy = some<CGaussianProcessRegression>();
	gpr_copy->load_serializable(file);
	file->close();

	file = some<CSerializableAsciiFile>(filename_predictions, 'r');
	auto predictions_copy = some<CRegressionLabels>();
	predictions_copy->load_serializable(file);

	/* now compare */
	set_global_fequals_epsilon(1e-10);
	EXPECT_TRUE(predictions->equals(predictions_copy));
	EXPECT_TRUE(gpr->equals(gpr_copy));
	set_global_fequals_epsilon(0);
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
		Y[i] = std::sin(X(0, i));
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
		Y[i] = std::sin(X(0, i));
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
	auto other_obj = some<CMockObject>();

	EXPECT_EQ(other_obj->ref_count(), 1);
	obj->put(Tag<CMockObject*>("watched_object"), other_obj.get());
	EXPECT_EQ(other_obj->ref_count(), 2);
	EXPECT_FALSE(other_obj->equals(obj));
	obj = empty<CMockObject>();
	EXPECT_EQ(other_obj->ref_count(), 1);
}

TEST(SGObject, watch_method)
{
	auto obj = some<CMockObject>();
	EXPECT_EQ(obj->get<int>("some_method"), obj->some_method());
	EXPECT_THROW(obj->put<int>("some_method", 0), ShogunException);
	EXPECT_NO_THROW(obj->to_string());
}

TEST(SGObject, subscribe_observer)
{
	auto obj = some<CMockObject>();
	auto param_obs = some<ParameterObserverScalar>();
	obj->subscribe_to_parameters(param_obs);

	EXPECT_EQ(obj->get<int64_t>("total_subscriptions"), 1);
}

TEST(SGObject, unsubscribe_observer)
{
	auto obj = some<CMockObject>();
	auto param_obs = some<ParameterObserverScalar>();
	auto subscription_index = obj->subscribe_to_parameters(param_obs);
	obj->unsubscribe(subscription_index);

	EXPECT_EQ(obj->get<int64_t>("total_subscriptions"), 0);
}

TEST(SGObject, unsubscribe_observer_failure)
{
	auto obj = some<CMockObject>();
	auto param_obs = some<ParameterObserverScalar>();
	obj->subscribe_to_parameters(param_obs);

	EXPECT_THROW(obj->unsubscribe(2), ShogunException);
}