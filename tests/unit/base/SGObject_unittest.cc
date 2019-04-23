/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Thoralf Klein, Wu Lin
 */

#include "sg_gtest_utilities.h"

#include "MockObject.h"
#include <shogun/base/ShogunEnv.h>
#include <shogun/base/class_list.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/stream/BufferedInputStream.h>
#include <shogun/io/stream/FileInputStream.h>
#include <shogun/io/stream/FileOutputStream.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/observers/ParameterObserverLogger.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/util/converters.h>
#include <shogun/mathematics/UniformRealDistribution.h>

#include <random>

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
	auto obj = std::make_shared<CloneEqualsMock<TypeParam>>();
}

TYPED_TEST(SGObjectEquals, same)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto obj2 = std::make_shared<CloneEqualsMock<TypeParam>>();

	EXPECT_TRUE(obj1->equals(obj1));
	EXPECT_TRUE(obj1->equals(obj2));
	EXPECT_TRUE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_null)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();

	EXPECT_FALSE(obj1->equals(nullptr));
}

TYPED_TEST(SGObjectEquals, different_basic)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto obj2 = std::make_shared<CloneEqualsMock<TypeParam>>();

	obj1->m_basic -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_string)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto obj2 = std::make_shared<CloneEqualsMock<TypeParam>>();

	obj1->m_string = "Oh no!";
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_object)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto obj2 = std::make_shared<CloneEqualsMock<TypeParam>>();

	obj1->m_object->m_some_value -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
	obj1->m_object->m_some_value += 1;

	obj1->m_object.reset();
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_sg_vector)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto obj2 = std::make_shared<CloneEqualsMock<TypeParam>>();

	obj1->m_sg_vector[0] -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_sg_sparse_vector)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto obj2 = std::make_shared<CloneEqualsMock<TypeParam>>();

	obj1->m_sg_sparse_vector.features[0].entry -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_sg_sparse_matrix)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto obj2 = std::make_shared<CloneEqualsMock<TypeParam>>();

	obj1->m_sg_sparse_matrix.sparse_matrix[0].features[0].entry -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_sg_matrix)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto obj2 = std::make_shared<CloneEqualsMock<TypeParam>>();

	obj1->m_sg_matrix(0, 0) = 0;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_raw_vector_basic)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto obj2 = std::make_shared<CloneEqualsMock<TypeParam>>();

	obj1->m_raw_vector_basic[0] -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_raw_vector_sg_vector)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto obj2 = std::make_shared<CloneEqualsMock<TypeParam>>();

	obj1->m_raw_vector_sg_vector[0].vector[0] -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_raw_vector_object)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto obj2 = std::make_shared<CloneEqualsMock<TypeParam>>();

	obj1->m_raw_vector_object[0]->m_some_value -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
	obj1->m_raw_vector_object[0]->m_some_value += 1;

	obj1->m_raw_vector_object[0].reset();
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TYPED_TEST(SGObjectEquals, different_raw_matrix_basic)
{
	auto obj1 = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto obj2 = std::make_shared<CloneEqualsMock<TypeParam>>();

	obj1->m_raw_matrix_basic[0] -= 1;
	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

TEST(SGObjectEquals, different_type)
{
	auto obj1 = std::make_shared<CloneEqualsMock<int>>();
	auto obj2 = std::make_shared<CloneEqualsMock<float>>();

	EXPECT_FALSE(obj1->equals(obj2));
	EXPECT_FALSE(obj2->equals(obj1));
}

/* FIXME
TYPED_TEST(SGObjectClone, basic_checks)
{
	auto obj = std::make_shared<CloneEqualsMock<TypeParam>>();

	auto clone = obj->clone();

	EXPECT_NE(clone, obj);
	ASSERT_NE(clone, nullptr);
	EXPECT_EQ(1, clone.use_count());

	auto clone_casted = clone->as<CloneEqualsMock<TypeParam>>();
	ASSERT_NE(clone_casted, nullptr);

	EXPECT_EQ(std::string(clone->get_name()), std::string(obj->get_name()));
}
*/

TYPED_TEST(SGObjectClone, equals_empty)
{
	auto obj = std::make_shared<CloneEqualsMock<TypeParam>>();

	auto clone = obj->clone();
	EXPECT_TRUE(clone->equals(obj));
}

TYPED_TEST(SGObjectClone, equals_non_empty)
{
	auto obj = std::make_shared<CloneEqualsMock<TypeParam>>();
	obj->m_basic -= 1;
	obj->m_string = "Non empty string";
	obj->m_object->m_some_value -= 1;
	obj->m_sg_vector[0] -= 1;
	obj->m_sg_matrix(0, 0) = 0;
	obj->m_sg_sparse_vector.features[0].entry -= 1;
	obj->m_sg_sparse_matrix.sparse_matrix[0].features[0].entry -= 1;
	obj->m_raw_vector_basic[0] -= 1;
	obj->m_raw_matrix_basic[0] -= 1;
	obj->m_raw_vector_sg_vector[0].vector[0] -= 1;
	obj->m_raw_vector_object[0]->m_some_value -= 1;

	auto clone = obj->clone();
	EXPECT_TRUE(clone->equals(obj));
}

/* FIXME
TYPED_TEST(SGObjectClone, not_just_copied_pointer)
{
	auto obj = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto clone_casted = obj->clone()->as<CloneEqualsMock< TypeParam > >();
	ASSERT_NE(clone_casted, nullptr);

	EXPECT_NE(clone_casted->m_object, obj->m_object);
	EXPECT_NE(clone_casted->m_raw_vector_basic, obj->m_raw_vector_basic);
	EXPECT_NE(clone_casted->m_raw_vector_sg_vector, obj->m_raw_vector_sg_vector);

	EXPECT_NE(clone_casted->m_raw_vector_object, obj->m_raw_vector_object);
	for (auto i : range(obj->m_raw_vector_object_len))
		EXPECT_NE(clone_casted->m_raw_vector_object[i], obj->m_raw_vector_object[i]);
}
*/

TYPED_TEST(SGObjectClone, equals_other_has_null_param)
{
	auto obj = std::make_shared<CloneEqualsMock<TypeParam>>();
	auto clone = obj->clone();
	EXPECT_TRUE(clone->equals(obj));
}

TEST(SGObject,equals_complex_equal)
{
	/* create some easy regression data: 1d noisy sine wave */
	int32_t seed = 12;
	index_t n=100;
	float64_t x_range=6;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	std::mt19937_64 prng(seed);
	UniformRealDistribution<float64_t> uniform_real_dist;
	for (index_t  i=0; i<n; ++i)
	{
		X[i]=uniform_real_dist(prng, {0.0, x_range});
		X_test[i]=(float64_t)i / n*x_range;
		Y[i] = std::sin(X[i]);
	}

	/* shogun representation */
	auto feat_train = std::make_shared<DenseFeatures<float64_t>>(X);
	auto feat_test = std::make_shared<DenseFeatures<float64_t>>(X_test);
	auto label_train = std::make_shared<RegressionLabels>(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	auto kernel = std::make_shared<GaussianKernel>(10, shogun_sigma);
	auto mean = std::make_shared<ZeroMean>();
	auto lik = std::make_shared<GaussianLikelihood>();
	lik->set_sigma(1);
	auto inf = std::make_shared<ExactInferenceMethod>(kernel, feat_train,
			mean, label_train, lik);
	auto gpr = std::make_shared<GaussianProcessRegression>(inf);

	/* train machine */
	gpr->train();

	// apply regression
	auto predictions=gpr->apply_regression(feat_test);

	/* save and load instance to compare */
	std::string filename_gpr {"gpr_instance.json"};
	std::string filename_predictions {"predictions_instance.json"};

	auto fs = env();
	ASSERT_TRUE(fs->file_exists(filename_gpr));
	std::unique_ptr<io::WritableFile> file;
	ASSERT_FALSE(fs->new_writable_file(filename_gpr, &file));
	auto fos = std::make_shared<io::FileOutputStream>(file.get());
	auto serializer = std::make_unique<io::JsonSerializer>();
	serializer->attach(fos);
	serializer->write(gpr);

	ASSERT_TRUE(fs->file_exists(filename_predictions));
	ASSERT_FALSE(fs->new_writable_file(filename_predictions, &file));
	fos = std::make_shared<io::FileOutputStream>(file.get());
	serializer->attach(fos);
	serializer->write(predictions);

	std::unique_ptr<io::RandomAccessFile> raf;
	ASSERT_FALSE(fs->new_random_access_file(filename_gpr, &raf));
	auto fis = std::make_unique<io::FileInputStream>(raf.get());
	auto bis = std::make_shared<io::BufferedInputStream>(fis.get());
	auto deserializer = std::make_unique<io::JsonDeserializer>();
	deserializer->attach(bis);
	auto gpr_copy = deserializer->read_object();

	ASSERT_FALSE(fs->new_random_access_file(filename_predictions, &raf));
	fis = std::make_unique<io::FileInputStream>(raf.get());
	bis = std::make_shared<io::BufferedInputStream>(fis.get());
	deserializer->attach(bis);
	auto predictions_copy = deserializer->read_object();

	/* now compare */
	env()->set_global_fequals_epsilon(1e-10);
	ASSERT_TRUE(predictions->equals(predictions_copy));
	ASSERT_TRUE(gpr->equals(gpr_copy));
	env()->set_global_fequals_epsilon(0);
	ASSERT_FALSE(fs->delete_file(filename_gpr));
	ASSERT_FALSE(fs->delete_file(filename_predictions));
}

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

	auto feat_train=std::make_shared<DenseFeatures<float64_t>>(X);
	auto label_train=std::make_shared<RegressionLabels>(Y);

	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	auto kernel=std::make_shared<GaussianKernel>(10, shogun_sigma);
	auto mean=std::make_shared<ZeroMean>();
	auto lik=std::make_shared<GaussianLikelihood>();
	lik->set_sigma(1);
	auto inf=std::make_shared<ExactInferenceMethod>(kernel, feat_train,
			mean, label_train, lik);

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
		Y[i] = std::sin(X(0, i));
	}

	auto feat_train=std::make_shared<DenseFeatures<float64_t>>(X);
	auto label_train=std::make_shared<RegressionLabels>(Y);

	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	auto kernel=std::make_shared<GaussianKernel>(10, shogun_sigma);
	auto mean=std::make_shared<ZeroMean>();
	auto lik=std::make_shared<GaussianLikelihood>();
	lik->set_sigma(1);
	auto inf=std::make_shared<ExactInferenceMethod>(kernel, feat_train,
			mean, label_train, lik);
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
	auto obj = std::make_shared<MockObject>();
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
	auto obj = std::make_shared<MockObject>();
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
	auto obj = std::make_shared<MockObject>();

	EXPECT_THROW(obj->get<int32_t>("foo"), ShogunException);
	obj->put("int", 10);
	EXPECT_EQ(obj->get(Tag<int32_t>("int")), 10);
	EXPECT_THROW(obj->get<float64_t>("int"), ShogunException);
	EXPECT_THROW(obj->get(Tag<float64_t>("int")), ShogunException);
	EXPECT_EQ(obj->get<int>("int"), 10);
}

TEST(SGObject, tags_has)
{
	auto obj = std::make_shared<MockObject>();

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
	auto obj = std::make_shared<MockObject>();

	obj->put("watched_int", 89);
	EXPECT_EQ(obj->get<int32_t>("watched_int"), 89);
	EXPECT_EQ(obj->get<int32_t>("watched_int"), obj->get_watched());
	obj->set_watched(12);
	EXPECT_EQ(obj->get<int32_t>("watched_int"), 12);
	EXPECT_EQ(obj->get<int32_t>("watched_int"), obj->get_watched());
}

TEST(SGObject, watch_method)
{
	auto obj = std::make_shared<MockObject>();
	EXPECT_EQ(obj->get<int>("some_method"), obj->some_method());
	EXPECT_THROW(obj->put<int>("some_method", 0), ShogunException);
	EXPECT_NO_THROW(obj->to_string());
}

TEST(SGObject, subscribe_observer)
{
	auto obj = std::make_shared<MockObject>();
	auto param_obs = std::make_shared<ParameterObserverLogger>();
	obj->subscribe(param_obs);

	EXPECT_EQ(param_obs->get<int64_t>("subscription_id"), 0);
	EXPECT_EQ(
	    obj->get<index_t>("num_subscriptions"),
	    utils::safe_convert<index_t>(1));
}

TEST(SGObject, unsubscribe_observer)
{
	auto obj = std::make_shared<MockObject>();
	auto param_obs = std::make_shared<ParameterObserverLogger>();
	obj->subscribe(param_obs);
	obj->unsubscribe(param_obs);

	EXPECT_EQ(param_obs->get<int64_t>("subscription_id"), -1);
	EXPECT_EQ(
	    obj->get<index_t>("num_subscriptions"),
	    utils::safe_convert<index_t>(0));
}

TEST(SGObject, unsubscribe_observer_failure)
{
	auto obj = std::make_shared<MockObject>();
	auto param_obs = std::make_shared<ParameterObserverLogger>();
	auto param_obs_not_in = std::make_shared<ParameterObserverLogger>();

	EXPECT_THROW(obj->unsubscribe(param_obs_not_in), ShogunException);
}

TEST(SGObject, constrained_parameter)
{
    auto obj = std::make_shared<MockObject>();
    obj->put("constrained_parameter", 1);
    EXPECT_EQ(obj->get<int32_t>("constrained_parameter"), 1);
    EXPECT_THROW(obj->put("constrained_parameter", 0), ShogunException);
    EXPECT_THROW(obj->put("constrained_parameter", 10), ShogunException);
    EXPECT_EQ(obj->get<int32_t>("constrained_parameter"), 1);
}

