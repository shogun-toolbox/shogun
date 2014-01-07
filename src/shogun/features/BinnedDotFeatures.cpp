/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#include <features/BinnedDotFeatures.h>
#include <base/Parameter.h>

using namespace shogun;

CBinnedDotFeatures::CBinnedDotFeatures(int32_t size)
	: CDotFeatures(size)
{
	init();
}


CBinnedDotFeatures::CBinnedDotFeatures(const CBinnedDotFeatures & orig)
	: CDotFeatures(orig), m_bins(orig.m_bins), m_fill(orig.m_fill),
	m_norm_one(orig.m_norm_one)
{
	init();
}

CBinnedDotFeatures::CBinnedDotFeatures(CDenseFeatures<float64_t>* sf, SGMatrix<float64_t> bins)
{
	init();
	set_simple_features(sf);
	set_bins(bins);

}

CBinnedDotFeatures::~CBinnedDotFeatures()
{
	SG_UNREF(m_features);
}

int32_t CBinnedDotFeatures::get_dim_feature_space() const
{
	return m_bins.num_rows*m_bins.num_cols;
}

float64_t CBinnedDotFeatures::dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())

	float64_t result=0;
	double sum1=0;
	double sum2=0;

	SGVector<float64_t> vec1=m_features->get_feature_vector(vec_idx1);
	SGVector<float64_t> vec2=((CBinnedDotFeatures*) df)->m_features->get_feature_vector(vec_idx2);

	for (int32_t i=0; i<m_bins.num_cols; i++)
	{
		float64_t v1=vec1.vector[i];
		float64_t v2=vec2.vector[i];
		float64_t* col=m_bins.get_column_vector(i);

		for (int32_t j=0; j<m_bins.num_rows; j++)
		{
			if (m_fill)
			{
				if (col[j]<=v1)
				{
					sum1+=1.0;

					if (col[j]<=v2)
					{
						sum2+=1.0;
						result+=1.0;
					}
				}
				else
				{
					if (col[j]<=v2)
						sum2+=1.0;
					else
						break;
				}

				/* the above is the fast version of
				if (col[j]<=v1 && col[j]<=v2)
					result+=1.0;

				if (col[j]<=v1)
					sum1+=1.0;

				if (col[j]<=v2)
					sum2+=1.0;
				*/
			}
			else
			{
				if (col[j]<=v1 && (j+1)<m_bins.num_rows && col[j+1]>v1 &&
						col[j]<=v2 && (j+1)<m_bins.num_rows && col[j+1]>v2)
				{
					result+=1;
					break;
				}
			}
		}
	}
	m_features->free_feature_vector(vec1, vec_idx1);
	((CBinnedDotFeatures*) df)->m_features->free_feature_vector(vec2, vec_idx2);

	if (m_fill && m_norm_one && sum1!=0 && sum2!=0)
		result/=CMath::sqrt(sum1*sum2);

	return result;

}

float64_t CBinnedDotFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	assert_shape(vec2_len);

	float64_t result=0;
	double sum=0;

	SGVector<float64_t> vec1=m_features->get_feature_vector(vec_idx1);


	for (int32_t i=0; i<m_bins.num_cols; i++)
	{
		float64_t v=vec1.vector[i];
		float64_t* col=m_bins.get_column_vector(i);
		int32_t offs=i*m_bins.num_rows;

		for (int32_t j=0; j<m_bins.num_rows; j++)
		{
			if (m_fill)
			{
				if (col[j]<=v)
				{
					result+=vec2[offs+j];
					sum+=1.0;
				}
			}
			else
			{
				if (col[j]<=v && (j+1)<m_bins.num_rows && col[j+1]>v)
				{
					result+=vec2[offs+j];
					break;
				}
			}
		}
	}
	m_features->free_feature_vector(vec1, vec_idx1);

	if (m_fill && m_norm_one && sum!=0)
		result/=CMath::sqrt(sum);

	return result;
}

void CBinnedDotFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	assert_shape(vec2_len);
	SGVector<float64_t> vec1=m_features->get_feature_vector(vec_idx1);

	if (m_fill && m_norm_one)
	{
		float64_t alpha_correction=0;
		for (int32_t i=0; i<m_bins.num_cols; i++)
		{
			float64_t v=vec1.vector[i];
			float64_t* col=m_bins.get_column_vector(i);

			for (int32_t j=0; j<m_bins.num_rows; j++)
			{
				if (col[j]<=v)
					alpha_correction+=1.0;
			}
		}

		if (alpha_correction==0.0)
			return;

		alpha/=CMath::sqrt(alpha_correction);
	}

	for (int32_t i=0; i<m_bins.num_cols; i++)
	{
		float64_t v=vec1.vector[i];
		float64_t* col=m_bins.get_column_vector(i);
		int32_t offs=i*m_bins.num_rows;

		for (int32_t j=0; j<m_bins.num_rows; j++)
		{
			if (m_fill)
			{
				if (col[j]<=v)
					vec2[offs+j]+=alpha;
			}
			else
			{
				if (col[j]<=v && (j+1)<m_bins.num_rows && col[j+1]>v)
				{
					vec2[offs+j]+=alpha;
					break;
				}
			}
		}
	}
	m_features->free_feature_vector(vec1, vec_idx1);
}

void CBinnedDotFeatures::assert_shape(int32_t vec2_len)
{
	if (m_bins.num_cols*m_bins.num_rows != vec2_len)
	{
		SG_ERROR("Bin matrix has shape (%d,%d) = %d entries, not matching vector"
				" length %d\n", m_bins.num_cols,m_bins.num_rows,
				m_bins.num_cols*m_bins.num_rows,vec2_len);
	}

	if (m_features && m_bins.num_cols != m_features->get_num_features())
	{
		SG_ERROR("Number of colums (%d) doesn't match number of features "
				"(%d)\n", m_bins.num_cols, m_features->get_num_features());
	}

}

int32_t CBinnedDotFeatures::get_nnz_features_for_vector(int32_t num)
{
	if (m_fill)
		return m_bins.num_rows;
	else
		return 1;
}

void* CBinnedDotFeatures::get_feature_iterator(int32_t vector_index)
{
	SG_NOTIMPLEMENTED
	return NULL;
}

bool CBinnedDotFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	SG_NOTIMPLEMENTED
	return false;
}

void CBinnedDotFeatures::free_feature_iterator(void* iterator)
{
	SG_NOTIMPLEMENTED
}

bool CBinnedDotFeatures::get_fill()
{
	return m_fill;
}

void CBinnedDotFeatures::set_fill(bool fill)
{
	m_fill=fill;
}

bool CBinnedDotFeatures::get_norm_one()
{
	return m_fill;
}

void CBinnedDotFeatures::set_norm_one(bool norm_one)
{
	m_norm_one=norm_one;
}

void CBinnedDotFeatures::set_bins(SGMatrix<float64_t> bins)
{
	m_bins=bins;
}

SGMatrix<float64_t> CBinnedDotFeatures::get_bins()
{
	return m_bins;
}

void CBinnedDotFeatures::set_simple_features(CDenseFeatures<float64_t>* features)
{
	SG_REF(features);
	m_features=features;
}

CDenseFeatures<float64_t>* CBinnedDotFeatures::get_simple_features()
{
	SG_REF(m_features);
	return m_features;
}

void CBinnedDotFeatures::init()
{
	m_features=NULL;
	m_fill=true;
	m_norm_one=false;
}

const char* CBinnedDotFeatures::get_name() const
{
	return "BinnedDotFeatures";
}

CFeatures* CBinnedDotFeatures::duplicate() const
{
	return new CBinnedDotFeatures(*this);
}

EFeatureType CBinnedDotFeatures::get_feature_type() const
{
	return F_DREAL;
}


EFeatureClass CBinnedDotFeatures::get_feature_class() const
{
	return C_BINNED_DOT;
}

int32_t CBinnedDotFeatures::get_num_vectors() const
{
	ASSERT(m_features)
	return m_features->get_num_vectors();
}
