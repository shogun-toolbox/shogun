#include <multiclass/MulticlassOneVsOneStrategy.h>
#include <multiclass/MulticlassOneVsRestStrategy.h>
#include <labels/BinaryLabels.h>
#include <labels/MulticlassLabels.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(MulticlassStrategy,rescale_ova_norm)
{
	SGVector<float64_t> scores(3);

	for (int32_t i=0; i<3; i++)
		scores[i] = (i+1)*0.1;

	CMulticlassOneVsRestStrategy ova(OVA_NORM);
	ova.set_num_classes(3);
	ova.rescale_outputs(scores);

	//SGVector<float64_t>::display_vector(scores.vector,scores.vlen);

	// GT caculated manually
	// scores[0] = scores[0] / sum(scores)
	// scores[1] = scores[1] / sum(scores)
	// scores[2] = scores[2] / sum(scores)
	EXPECT_NEAR(scores[0],0.16666666666666669,1E-5);
	EXPECT_NEAR(scores[1],0.33333333333333337,1E-5);
	EXPECT_NEAR(scores[2],0.5,1E-5);
}

TEST(MulticlassStrategy,rescale_ova_softmax)
{
	SGVector<float64_t> scores(3);
	scores.range_fill(1);

	SGVector<float64_t> As(3);
	SGVector<float64_t> Bs(3);
	Bs.zero();
	for (int32_t i=0; i<3; i++)
		As[i] = (i+1)*0.1;

	CMulticlassOneVsRestStrategy ova(OVA_SOFTMAX);
	ova.set_num_classes(3);
	ova.rescale_outputs(scores,As,Bs);

	//SGVector<float64_t>::display_vector(scores.vector,scores.vlen);

	// GT caculated manually
	// scores[0] = exp(-0.1) / norm
	// scores[1] = exp(-0.4) / norm
	// scores[2] = exp(-0.9) / norm
	// norm = exp(-0.1)+exp(-0.4)+exp(-0.9)
	EXPECT_NEAR(scores[0],0.4565903181944378,1E-5);
	EXPECT_NEAR(scores[1],0.33825042710530284,1E-5);
	EXPECT_NEAR(scores[2],0.20515925470025934,1E-5);
}

TEST(MulticlassStrategy,rescale_ova_price)
{
	SGVector<float64_t> scores(3);
	SGVector<float64_t>::fill_vector(scores.vector,scores.vlen,0.5);

	CMulticlassOneVsOneStrategy ovo(OVO_PRICE);
	ovo.set_num_classes(3);
	ovo.rescale_outputs(scores);

	//SGVector<float64_t>::display_vector(scores.vector,scores.vlen);

	// GT caculated manually
	// scores[0] = \frac{1}{1/0.5+1/0.5-(3-2)} / norm
	// scores[1] = \frac{1}{1/0.5+1/0.5-(3-2)} / norm
	// scores[2] = \frac{1}{1/0.5+1/0.5-(3-2)} / norm
	// norm = sum(scores)
	EXPECT_NEAR(scores[0],0.3333333333333333,1E-5);
	EXPECT_NEAR(scores[1],0.3333333333333333,1E-5);
	EXPECT_NEAR(scores[2],0.3333333333333333,1E-5);
}

TEST(MulticlassStrategy,rescale_ova_hastie)
{
	CMulticlassOneVsOneStrategy ovo(OVO_HASTIE);
	ovo.set_num_classes(3);

	// training simulation
	SGVector<float64_t> labels(3);
	labels.range_fill(0);

	CMulticlassLabels *orig_labels = new CMulticlassLabels(labels);
	SG_REF(orig_labels);

	CBinaryLabels *train_labels = new CBinaryLabels(2);
	SG_REF(train_labels);

	ovo.train_start(orig_labels, train_labels);
	for (int32_t i=0; i<3; i++)
	{
		ovo.train_prepare_next();
	}
	ovo.train_stop();

	SGVector<float64_t> scores(3);
	SGVector<float64_t>::fill_vector(scores.vector,scores.vlen,0.5);

	ovo.rescale_outputs(scores);

	//SGVector<float64_t>::display_vector(scores.vector,scores.vlen);

	EXPECT_NEAR(scores[0],0.3333333333333333,1E-5);
	EXPECT_NEAR(scores[1],0.3333333333333333,1E-5);
	EXPECT_NEAR(scores[2],0.3333333333333333,1E-5);

	SG_UNREF(orig_labels);
	SG_UNREF(train_labels);
}

TEST(MulticlassStrategy,rescale_ova_hamamura)
{
	SGVector<float64_t> scores(3);
	SGVector<float64_t>::fill_vector(scores.vector,scores.vlen,0.5);

	CMulticlassOneVsOneStrategy ovo(OVO_HAMAMURA);
	ovo.set_num_classes(3);
	ovo.rescale_outputs(scores);

	//SGVector<float64_t>::display_vector(scores.vector,scores.vlen);

	EXPECT_NEAR(scores[0],0.3333333333333333,1E-5);
	EXPECT_NEAR(scores[1],0.3333333333333333,1E-5);
	EXPECT_NEAR(scores[2],0.3333333333333333,1E-5);
}
