/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2009 Soeren Sonnnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/AUCKernel.h>
#include <shogun/io/SGIO.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

void
CAUCKernel::init()
{
	SG_ADD((CSGObject**) &subkernel, "subkernel", "The subkernel.",
	    MS_AVAILABLE);
}

CAUCKernel::CAUCKernel()
: CDotKernel(0), subkernel(NULL)
{
	init();
}

CAUCKernel::CAUCKernel(int32_t size, CKernel* s)
: CDotKernel(size), subkernel(s)
{
	init();
	SG_REF(subkernel);
}

CAUCKernel::~CAUCKernel()
{
	SG_UNREF(subkernel);
	cleanup();
}

CLabels* CAUCKernel::setup_auc_maximization(CLabels* labels)
{
	SG_INFO("setting up AUC maximization\n")
	ASSERT(labels)
	ASSERT(labels->get_label_type() == LT_BINARY)
	labels->ensure_valid();

	// get the original labels
	SGVector<int32_t> int_labels=((CBinaryLabels*) labels)->get_int_labels();
	ASSERT(subkernel->get_num_vec_rhs()==int_labels.vlen)

	// count positive and negative
	int32_t num_pos=0;
	int32_t num_neg=0;

	for (int32_t i=0; i<int_labels.vlen; i++)
	{
		if (int_labels.vector[i]==1)
			num_pos++;
		else
			num_neg++;
	}

	// create AUC features and labels (alternate labels)
	int32_t num_auc = num_pos*num_neg;
	SG_INFO("num_pos: %i  num_neg: %i  num_auc: %i\n", num_pos, num_neg, num_auc)

	SGMatrix<uint16_t> features_auc(2,num_auc);
	int32_t* labels_auc = SG_MALLOC(int32_t, num_auc);
	int32_t n=0 ;

	for (int32_t i=0; i<int_labels.vlen; i++)
	{
		if (int_labels.vector[i]!=1)
			continue;

		for (int32_t j=0; j<int_labels.vlen; j++)
		{
			if (int_labels.vector[j]!=-1)
				continue;

			// create about as many positively as negatively labeled examples
			if (n%2==0)
			{
				features_auc.matrix[n*2]=i;
				features_auc.matrix[n*2+1]=j;
				labels_auc[n]=1;
			}
			else
			{
				features_auc.matrix[n*2]=j;
				features_auc.matrix[n*2+1]=i;
				labels_auc[n]=-1;
			}

			n++;
			ASSERT(n<=num_auc)
		}
	}

	// create label object and attach it to svm
	CBinaryLabels* lab_auc = new CBinaryLabels(num_auc);
	lab_auc->set_int_labels(SGVector<int32_t>(labels_auc, num_auc, false));
	SG_REF(lab_auc);

	// create feature object
	CDenseFeatures<uint16_t>* f = new CDenseFeatures<uint16_t>(0);
	f->set_feature_matrix(features_auc);

	// create AUC kernel and attach the features
	init(f,f);

	SG_FREE(labels_auc);

	return lab_auc;
}


bool CAUCKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);
	init_normalizer();
	return true;
}

float64_t CAUCKernel::compute(int32_t idx_a, int32_t idx_b)
{
  int32_t alen, blen;
  bool afree, bfree;

  uint16_t* avec=((CDenseFeatures<uint16_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
  uint16_t* bvec=((CDenseFeatures<uint16_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);

  ASSERT(alen==2)
  ASSERT(blen==2)

  ASSERT(subkernel && subkernel->has_features())

  float64_t k11,k12,k21,k22;
  int32_t idx_a1=avec[0], idx_a2=avec[1], idx_b1=bvec[0], idx_b2=bvec[1];

  k11 = subkernel->kernel(idx_a1,idx_b1);
  k12 = subkernel->kernel(idx_a1,idx_b2);
  k21 = subkernel->kernel(idx_a2,idx_b1);
  k22 = subkernel->kernel(idx_a2,idx_b2);

  float64_t result = k11+k22-k21-k12;

  ((CDenseFeatures<uint16_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CDenseFeatures<uint16_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
