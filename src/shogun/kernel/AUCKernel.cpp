/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Chiyuan Zhang, Bjoern Esser,
 *          Sergey Lisitsyn
 */

#include <shogun/io/SGIO.h>
#include <shogun/kernel/AUCKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

void AUCKernel::init()
{
	SG_ADD(
	    &subkernel, "subkernel", "The subkernel.", ParameterProperties::HYPER);
	SG_ADD(&labels, "labels", "The labels.");
	watch_method("setup_auc_maximization", &AUCKernel::setup_auc_maximization);
}

AUCKernel::AUCKernel() : DotKernel(0), subkernel(nullptr), labels(nullptr)
{
	init();
}

AUCKernel::AUCKernel(int32_t size, std::shared_ptr<Kernel> s, std::shared_ptr<Labels> l)
    : DotKernel(size), subkernel(s), labels(l)
{
	init();
}

AUCKernel::~AUCKernel()
{
	cleanup();
}

bool AUCKernel::setup_auc_maximization()
{
	io::info("setting up AUC maximization");
	ASSERT(labels)
	ASSERT(labels->get_label_type() == LT_BINARY)
	labels->ensure_valid();

	// get the original labels
	SGVector<int32_t> int_labels = binary_labels(labels)->get_int_labels();

	ASSERT(subkernel->get_num_vec_rhs() == int_labels.vlen)

	// count positive and negative
	int32_t num_pos = 0;
	int32_t num_neg = 0;

	for (int32_t i = 0; i < int_labels.vlen; i++)
	{
		if (int_labels.vector[i] == 1)
			num_pos++;
		else
			num_neg++;
	}

	// create AUC features and labels (alternate labels)
	int32_t num_auc = num_pos * num_neg;
	io::info(
	    "num_pos: {}  num_neg: {}  num_auc: {}", num_pos, num_neg, num_auc);

	SGMatrix<uint16_t> features_auc(2, num_auc);
	auto* labels_auc = SG_MALLOC(int32_t, num_auc);
	int32_t n = 0;

	for (int32_t i = 0; i < int_labels.vlen; i++)
	{
		if (int_labels.vector[i] != 1)
			continue;

		for (int32_t j = 0; j < int_labels.vlen; j++)
		{
			if (int_labels.vector[j] != -1)
				continue;

			// create about as many positively as negatively labeled examples
			if (n % 2 == 0)
			{
				features_auc.matrix[n * 2] = i;
				features_auc.matrix[n * 2 + 1] = j;
				labels_auc[n] = 1;
			}
			else
			{
				features_auc.matrix[n * 2] = j;
				features_auc.matrix[n * 2 + 1] = i;
				labels_auc[n] = -1;
			}

			n++;
			ASSERT(n <= num_auc)
		}
	}

	// create label object and attach it to svm
	auto lab_auc = std::make_shared<BinaryLabels>(num_auc);
	lab_auc->set_int_labels(SGVector<int32_t>(labels_auc, num_auc, false));


	// create feature object
	auto f = std::make_shared<DenseFeatures<uint16_t>>(features_auc);

	// create AUC kernel and attach the features
	init(f, f);

	SG_FREE(labels_auc);

	return true;
}

bool AUCKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	DotKernel::init(l, r);
	init_normalizer();
	return true;
}

float64_t AUCKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	uint16_t* avec=lhs->as<DenseFeatures<uint16_t>>()->get_feature_vector(idx_a, alen, afree);
	uint16_t* bvec=rhs->as<DenseFeatures<uint16_t>>()->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen == 2)
	ASSERT(blen == 2)

	ASSERT(subkernel && subkernel->has_features())

	float64_t k11, k12, k21, k22;
	int32_t idx_a1 = avec[0], idx_a2 = avec[1], idx_b1 = bvec[0],
	        idx_b2 = bvec[1];

	k11 = subkernel->kernel(idx_a1, idx_b1);
	k12 = subkernel->kernel(idx_a1, idx_b2);
	k21 = subkernel->kernel(idx_a2, idx_b1);
	k22 = subkernel->kernel(idx_a2, idx_b2);

	float64_t result = k11 + k22 - k21 - k12;
	lhs->as<DenseFeatures<uint16_t>>()->free_feature_vector(avec, idx_a, afree);
	rhs->as<DenseFeatures<uint16_t>>()->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
