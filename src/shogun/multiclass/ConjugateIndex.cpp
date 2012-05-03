/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Sergey Lisitsyn
 */

#include <shogun/multiclass/ConjugateIndex.h>
#ifdef HAVE_LAPACK
#include <shogun/machine/Machine.h>
#include <shogun/features/Features.h>
#include <shogun/features/Labels.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CConjugateIndex::CConjugateIndex() : CMachine()
{
	m_classes = NULL;
	m_features = NULL;
};

CConjugateIndex::CConjugateIndex(CFeatures* train_features, CLabels* train_labels) : CMachine()
{
	m_features = NULL;
	set_features(train_features);
	set_labels(train_labels);
	m_classes = NULL;
};

CConjugateIndex::~CConjugateIndex()
{
	clean_classes();
	SG_UNREF(m_features);
};

void CConjugateIndex::set_features(CFeatures* features)
{
	ASSERT(features->get_feature_class()==C_SIMPLE);
	SG_REF(features);
	SG_UNREF(m_features);
	m_features = (CSimpleFeatures<float64_t>*)features;
}

CSimpleFeatures<float64_t>* CConjugateIndex::get_features()
{
	SG_REF(m_features);
	return m_features;
}

void CConjugateIndex::clean_classes()
{
	if (m_classes)
	{
		for (int32_t i=0; i<m_num_classes; i++)
			m_classes[i].destroy_matrix();

		delete[] m_classes;
	}
}

bool CConjugateIndex::train(CFeatures* train_features)
{
	if (train_features)
		set_features(train_features);

	m_num_classes = m_labels->get_num_classes();
	ASSERT(m_num_classes>=2);
	clean_classes();

	int32_t num_vectors;
	int32_t num_features;
	float64_t* feature_matrix = m_features->get_feature_matrix(num_features,num_vectors);

	m_classes = new SGMatrix<float64_t>[m_num_classes];
	for (int32_t i=0; i<m_num_classes; i++)
		m_classes[i] = SGMatrix<float64_t>(num_features,num_features);

	m_feature_vector = SGVector<float64_t>(num_features);

	SG_PROGRESS(0,0,m_num_classes-1);

	for (int32_t label=0; label<m_num_classes; label++)
	{
		int32_t count = 0;
		for (int32_t i=0; i<num_vectors; i++)
		{
			if (m_labels->get_int_label(i) == label)
				count++;
		}

		SGMatrix<float64_t> class_feature_matrix(num_features,count);
		SGMatrix<float64_t> matrix(count,count);
		SGMatrix<float64_t> helper_matrix(num_features,count);

		count = 0;
		for (int32_t i=0; i<num_vectors; i++)
		{
			if (m_labels->get_label(i) == label)
			{
				memcpy(class_feature_matrix.matrix+count*num_features,
				       feature_matrix+i*num_features,
				       sizeof(float64_t)*num_features);
				count++;
			}
		}

		cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,
		            count,count,num_features,
		            1.0,class_feature_matrix.matrix,num_features,
		            class_feature_matrix.matrix,num_features,
		            0.0,matrix.matrix,count);

		CMath::inverse(SGMatrix<float64_t>(matrix.matrix,count,count));

		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,
		            count,num_features,count,
		            1.0,matrix.matrix,count,
		            class_feature_matrix.matrix,num_features,
		            0.0,helper_matrix.matrix,count);

		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
		            num_features,num_features,count,
		            1.0,class_feature_matrix.matrix,num_features,
		            helper_matrix.matrix,count,
		            0.0,m_classes[label].matrix,num_features);

		SG_PROGRESS(label+1,0,m_num_classes);
		helper_matrix.destroy_matrix();
		class_feature_matrix.destroy_matrix();
		matrix.destroy_matrix();
	}
	SG_DONE();

	return true;
};

CLabels* CConjugateIndex::apply(CFeatures* test_features)
{
	set_features(test_features);

	CLabels* predicted_labels = apply();

	return predicted_labels;
};

CLabels* CConjugateIndex::apply()
{
	ASSERT(m_classes);
	ASSERT(m_num_classes>1);
	ASSERT(m_features->get_num_features()==m_feature_vector.vlen);

	int32_t num_vectors = m_features->get_num_vectors();

	CLabels* predicted_labels = new CLabels(num_vectors);

	for (int32_t i=0; i<num_vectors;i++)
	{
		SG_PROGRESS(i,0,num_vectors-1);
		predicted_labels->set_label(i,apply(i));
	}
	SG_DONE();

	return predicted_labels;
};

float64_t CConjugateIndex::conjugate_index(const SGVector<float64_t>& feature_vector, int32_t label)
{
	int32_t num_features = feature_vector.vlen;
	float64_t norm = cblas_ddot(num_features,feature_vector.vector,1,
	                            feature_vector.vector,1);

	cblas_dgemv(CblasColMajor,CblasNoTrans,
	            num_features,num_features,
	            1.0,m_classes[label].matrix,num_features,
	            feature_vector.vector,1,
	            0.0,m_feature_vector.vector,1);

	float64_t product = cblas_ddot(num_features,feature_vector.vector,1,
	                               m_feature_vector.vector,1);
	return product/norm;
};

float64_t CConjugateIndex::apply(int32_t index)
{
	int32_t predicted_label = 0;
	float64_t max_conjugate_index = 0.0;
	float64_t current_conjugate_index;

	SGVector<float64_t> feature_vector = m_features->get_feature_vector(index);
	for (int32_t i=0; i<m_num_classes; i++)
	{
		current_conjugate_index = conjugate_index(feature_vector,i);

		if (current_conjugate_index > max_conjugate_index)
		{
			max_conjugate_index = current_conjugate_index;
			predicted_label = i;
		}
	}

	return predicted_label;
};

#endif /* HAVE_LAPACK */
