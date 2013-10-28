#include "MockLatentModel.h"
#include <shogun/lib/config.h>
#include <shogun/latent/LatentSVM.h>

using namespace shogun;
using ::testing::Return;
using ::testing::NiceMock;

/** gmock REV 443 and freebsd doesn't play nicely */
#ifdef FREEBSD
TEST(LatentModel, DISABLED_argmax_h)
#else
TEST(LatentModel, argmax_h)
#endif
{
	using ::testing::AtMost;
	using ::testing::_;
	using ::testing::NiceMock;

	NiceMock<MockCLatentModel> model;
	int32_t dim = 10, samples = 20;
	SGVector<float64_t> a(dim);
	CData* data = new CData();

	ON_CALL(model, get_dim())
		.WillByDefault(Return(dim));

	ON_CALL(model, get_num_vectors())
		.WillByDefault(Return(samples));

	EXPECT_CALL(model, infer_latent_variable(_,_))
		.Times(AtMost(samples))
		.WillRepeatedly(Return(data));

	model.argmax_h(a);

	SG_UNREF(data);
}

#ifdef USE_REFERENCE_COUNTING

#ifdef FREEBSD
TEST(LatentSVM, DISABLED_ctor)
#else
TEST(LatentSVM, ctor)
#endif
{
	using ::testing::AtLeast;
	using ::testing::Exactly;

	MockCLatentModel model; model.ref();
	int32_t dim = 10, samples = 20;

	ON_CALL(model, get_dim())
		.WillByDefault(Return(dim));

	ON_CALL(model, get_num_vectors())
		.WillByDefault(Return(samples));

	EXPECT_CALL(model, get_dim())
		.Times(Exactly(1));

	CLatentSVM* lsvm = new CLatentSVM(&model, 10);

	SG_UNREF(lsvm);
}

#ifdef FREEBSD
TEST(LatentSVM, DISABLED_apply)
#else
TEST(LatentSVM, apply)
#endif
{
	using ::testing::AtMost;
	using ::testing::_;
	using ::testing::NiceMock;

	NiceMock<MockCLatentModel> model; model.ref();
	int32_t dim = 10, samples = 20;
	SGMatrix<float64_t> feats(dim, samples);
	CDenseFeatures<float64_t>* dense_feats = new CDenseFeatures<float64_t>(feats);
	CData* data = new CData();
	CLatentFeatures* f = new CLatentFeatures(samples);

	ON_CALL(model, get_dim())
		.WillByDefault(Return(dim));

	ON_CALL(model, get_num_vectors())
		.WillByDefault(Return(samples));

	EXPECT_CALL(model, get_num_vectors())
		.Times(2);

	EXPECT_CALL(model, infer_latent_variable(_,_))
		.Times(samples)
		.WillRepeatedly(Return(data));

	EXPECT_CALL(model, get_psi_feature_vectors())
		.Times(1)
		.WillOnce(Return(dense_feats));


	CLatentSVM* lsvm = new CLatentSVM(&model, 10);

	lsvm->apply(f);

	SG_UNREF(lsvm);
	SG_UNREF(dense_feats);
}
#endif
