/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Vladislav Horbatiuk, Evgeniy Andreev,
 *          Evan Shelhamer, Sergey Lisitsyn
 */

#include <shogun/features/BinnedDotFeatures.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

BinnedDotFeatures::BinnedDotFeatures(int32_t size)
	: DotFeatures(size)
{
	init();
}


BinnedDotFeatures::BinnedDotFeatures(const BinnedDotFeatures & orig)
	: DotFeatures(orig), m_bins(orig.m_bins), m_fill(orig.m_fill),
	m_norm_one(orig.m_norm_one)
{
	init();
}

BinnedDotFeatures::BinnedDotFeatures(std::shared_ptr<Features> sf, SGMatrix<float64_t> bins)
{
	init();
	set_simple_features(std::static_pointer_cast<DenseFeatures<float64_t>>(sf));
	set_bins(bins);

}

BinnedDotFeatures::~BinnedDotFeatures()
{

}

int32_t BinnedDotFeatures::get_dim_feature_space() const
{
	return m_bins.num_rows*m_bins.num_cols;
}

float64_t BinnedDotFeatures::dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())

	float64_t result=0;
	double sum1=0;
	double sum2=0;

	auto bin_dots = std::static_pointer_cast<BinnedDotFeatures>(df);
	SGVector<float64_t> vec1=m_features->get_feature_vector(vec_idx1);
	SGVector<float64_t> vec2=bin_dots->m_features->get_feature_vector(vec_idx2);

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
	bin_dots->m_features->free_feature_vector(vec2, vec_idx2);

	if (m_fill && m_norm_one && sum1!=0 && sum2!=0)
		result /= std::sqrt(sum1 * sum2);

	return result;

}

float64_t
BinnedDotFeatures::dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const
{
	assert_shape(vec2.size());

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
		result /= std::sqrt(sum);

	return result;
}

void BinnedDotFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val) const
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

		alpha /= std::sqrt(alpha_correction);
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

void BinnedDotFeatures::assert_shape(int32_t vec2_len) const
{
	if (m_bins.num_cols*m_bins.num_rows != vec2_len)
	{
		error("Bin matrix has shape ({},{}) = {} entries, not matching vector"
				" length {}", m_bins.num_cols,m_bins.num_rows,
				m_bins.num_cols*m_bins.num_rows,vec2_len);
	}

	if (m_features && m_bins.num_cols != m_features->get_num_features())
	{
		error("Number of colums ({}) doesn't match number of features "
				"({})", m_bins.num_cols, m_features->get_num_features());
	}

}

int32_t BinnedDotFeatures::get_nnz_features_for_vector(int32_t num) const
{
	if (m_fill)
		return m_bins.num_rows;
	else
		return 1;
}

void* BinnedDotFeatures::get_feature_iterator(int32_t vector_index)
{
	not_implemented(SOURCE_LOCATION);
	return NULL;
}

bool BinnedDotFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	not_implemented(SOURCE_LOCATION);
	return false;
}

void BinnedDotFeatures::free_feature_iterator(void* iterator)
{
	not_implemented(SOURCE_LOCATION);
}

bool BinnedDotFeatures::get_fill()
{
	return m_fill;
}

void BinnedDotFeatures::set_fill(bool fill)
{
	m_fill=fill;
}

bool BinnedDotFeatures::get_norm_one()
{
	return m_fill;
}

void BinnedDotFeatures::set_norm_one(bool norm_one)
{
	m_norm_one=norm_one;
}

void BinnedDotFeatures::set_bins(SGMatrix<float64_t> bins)
{
	m_bins=bins;
}

SGMatrix<float64_t> BinnedDotFeatures::get_bins()
{
	return m_bins;
}

void BinnedDotFeatures::set_simple_features(std::shared_ptr<DenseFeatures<float64_t>> features)
{
	m_features=features;
}

std::shared_ptr<DenseFeatures<float64_t>> BinnedDotFeatures::get_simple_features()
{
	return m_features;
}

void BinnedDotFeatures::init()
{
	m_features=NULL;
	m_fill=true;
	m_norm_one=false;
}

const char* BinnedDotFeatures::get_name() const
{
	return "BinnedDotFeatures";
}

std::shared_ptr<Features> BinnedDotFeatures::duplicate() const
{
	return std::make_shared<BinnedDotFeatures>(*this);
}

EFeatureType BinnedDotFeatures::get_feature_type() const
{
	return F_DREAL;
}


EFeatureClass BinnedDotFeatures::get_feature_class() const
{
	return C_BINNED_DOT;
}

int32_t BinnedDotFeatures::get_num_vectors() const
{
	ASSERT(m_features)
	return m_features->get_num_vectors();
}
