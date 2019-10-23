/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Vladislav Horbatiuk,
 *          Evan Shelhamer
 */

#include <shogun/features/WDFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

WDFeatures::WDFeatures() :DotFeatures()
{
	unstable(SOURCE_LOCATION);

	strings = NULL;

	degree = 0;
	from_degree = 0;
	string_length = 0;
	num_strings = 0;
	alphabet_size = 0;
	w_dim = 0;
	wd_weights = NULL;
	normalization_const = 0.0;
}

WDFeatures::WDFeatures(const std::shared_ptr<StringFeatures<uint8_t>>& str,
		int32_t order, int32_t from_order) : DotFeatures()
{
	ASSERT(str)
	ASSERT(str->have_same_length())


	strings=str;
	string_length=str->get_max_vector_length();
	num_strings=str->get_num_vectors();
	auto alpha=str->get_alphabet();
	alphabet_size=alpha->get_num_symbols();

	degree=order;
	from_degree=from_order;
	wd_weights=NULL;
	set_wd_weights();
	set_normalization_const();

}

WDFeatures::WDFeatures(const WDFeatures& orig)
	: DotFeatures(orig), strings(orig.strings),
	degree(orig.degree), from_degree(orig.from_degree),
	normalization_const(orig.normalization_const)
{


	if (strings)
	{
		string_length=strings->get_max_vector_length();
		num_strings=strings->get_num_vectors();
		auto alpha=strings->get_alphabet();
		alphabet_size=alpha->get_num_symbols();
	}
	else
	{
		string_length = 0;
		num_strings = 0;
		alphabet_size = 0;
	}

	wd_weights=NULL;
	if (degree>0)
		set_wd_weights();
}

WDFeatures::~WDFeatures()
{

	SG_FREE(wd_weights);
}

float64_t WDFeatures::dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	auto wdf = std::static_pointer_cast<WDFeatures>(df);

	int32_t len1, len2;
	bool free_vec1, free_vec2;

	uint8_t* vec1=strings->get_feature_vector(vec_idx1, len1, free_vec1);
	uint8_t* vec2=wdf->strings->get_feature_vector(vec_idx2, len2, free_vec2);

	ASSERT(len1==len2)

	float64_t sum=0.0;

	for (int32_t i=0; i<len1; i++)
	{
		for (int32_t j=0; (i+j<len1) && (j<degree); j++)
		{
			if (vec1[i+j]!=vec2[i+j])
				break ;
			sum += wd_weights[j]*wd_weights[j];
		}
	}
	strings->free_feature_vector(vec1, vec_idx1, free_vec1);
	wdf->strings->free_feature_vector(vec2, vec_idx2, free_vec2);
	return sum/Math::sq(normalization_const);
}

float64_t
WDFeatures::dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const
{
	require(
	    vec2.size() == w_dim, "Dimensions don't match, vec2_dim={}, w_dim={}",
	    vec2.size(), w_dim);

	float64_t sum=0;
	int32_t lim=Math::min(degree, string_length);
	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	int32_t* val=SG_MALLOC(int32_t, len);
	SGVector<int32_t>::fill_vector(val, len, 0);

	int32_t asize=alphabet_size;
	int32_t asizem1=1;
	int32_t offs=0;

	for (int32_t k=0; k<lim; k++)
	{
		float64_t wd = wd_weights[k];

		int32_t o=offs;
		for (int32_t i=0; i+k < len; i++)
		{
			val[i]+=asizem1*vec[i+k];
			sum+=vec2[val[i]+o]*wd;
			o+=asize;
		}
		offs+=asize*len;
		asize*=alphabet_size;
		asizem1*=alphabet_size;
	}
	SG_FREE(val);
	strings->free_feature_vector(vec, vec_idx1, free_vec1);

	return sum/normalization_const;
}

void WDFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val) const
{
	if (vec2_len != w_dim)
		error("Dimensions don't match, vec2_dim={}, w_dim={}", vec2_len, w_dim);

	int32_t lim=Math::min(degree, string_length);
	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	int32_t* val=SG_MALLOC(int32_t, len);
	SGVector<int32_t>::fill_vector(val, len, 0);

	int32_t asize=alphabet_size;
	int32_t asizem1=1;
	int32_t offs=0;

	for (int32_t k=0; k<lim; k++)
	{
		float64_t wd = alpha*wd_weights[k]/normalization_const;

		if (abs_val)
			wd=Math::abs(wd);

		int32_t o=offs;
		for (int32_t i=0; i+k < len; i++)
		{
			val[i]+=asizem1*vec[i+k];
			vec2[val[i]+o]+=wd;
			o+=asize;
		}
		offs+=asize*len;
		asize*=alphabet_size;
		asizem1*=alphabet_size;
	}
	SG_FREE(val);

	strings->free_feature_vector(vec, vec_idx1, free_vec1);
}

void WDFeatures::set_wd_weights()
{
	ASSERT(degree>0 && degree<=8)
	SG_FREE(wd_weights);
	wd_weights=SG_MALLOC(float64_t, degree);
	w_dim=0;

	for (int32_t i=0; i<degree; i++)
	{
		w_dim+=Math::pow(alphabet_size, i+1)*string_length;
		wd_weights[i]=sqrt(2.0*(from_degree-i)/(from_degree*(from_degree+1)));
	}
	SG_DEBUG("created WDFeatures with d={} ({}), alphabetsize={}, dim={} num={}, len={}", degree, from_degree, alphabet_size, w_dim, num_strings, string_length)
}


void WDFeatures::set_normalization_const(float64_t n)
{
	if (n==0)
	{
		normalization_const=0;
		for (int32_t i=0; i<degree; i++)
			normalization_const+=(string_length-i)*wd_weights[i]*wd_weights[i];

		normalization_const = std::sqrt(normalization_const);
	}
	else
		normalization_const=n;

	SG_DEBUG("normalization_const:{}", normalization_const)
}

void* WDFeatures::get_feature_iterator(int32_t vector_index)
{
	if (vector_index>=num_strings)
	{
		error("Index out of bounds (number of strings {}, you "
				"requested {})", num_strings, vector_index);
	}

	wd_feature_iterator* it=SG_MALLOC(wd_feature_iterator, 1);

	it->lim=Math::min(degree, string_length);
	it->vec= strings->get_feature_vector(vector_index, it->vlen, it->vfree);
	it->vidx=vector_index;

	it->vec = strings->get_feature_vector(vector_index, it->vlen, it->vfree);
	it->val=SG_MALLOC(int32_t, it->vlen);
	SGVector<int32_t>::fill_vector(it->val, it->vlen, 0);

	it->asize=alphabet_size;
	it->asizem1=1;
	it->offs=0;
	it->k=0;
	it->i=0;
	it->o=0;

	return it;
}

bool WDFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	wd_feature_iterator* it=(wd_feature_iterator*) iterator;

	if (it->i + it->k >= it->vlen)
	{
		if (it->k < it->lim-1)
		{
			it->offs+=it->asize*it->vlen;
			it->asize*=alphabet_size;
			it->asizem1*=alphabet_size;
			it->k++;
			it->i=0;
			it->o=it->offs;
		}
		else
			return false;
	}

	int32_t i=it->i;
	int32_t k=it->k;
#ifdef DEBUG_WDFEATURES
	io::print("i={} k={} offs={} o={} asize={} asizem1={}\n", i, k, it->offs, it->o, it->asize, it->asizem1);
#endif

	it->val[i]+=it->asizem1*it->vec[i+k];
	value=wd_weights[k]/normalization_const;
	index=it->val[i]+it->o;
#ifdef DEBUG_WDFEATURES
	io::print("index={} val={} w_size={} lim={} vlen={}\n", index, value, w_dim, it->lim, it->vlen);
#endif

	it->o+=it->asize;
	it->i=i+1;

	return true;
}

void WDFeatures::free_feature_iterator(void* iterator)
{
	ASSERT(iterator)
	wd_feature_iterator* it=(wd_feature_iterator*) iterator;
	strings->free_feature_vector(it->vec, it->vidx, it->vfree);
	SG_FREE(it->val);
	SG_FREE(it);
}

std::shared_ptr<Features> WDFeatures::duplicate() const
{
	return std::make_shared<WDFeatures>(*this);
}

int32_t WDFeatures::get_dim_feature_space() const
{
	return w_dim;
}

int32_t WDFeatures::get_nnz_features_for_vector(int32_t num) const
{
	int32_t vlen=-1;
	bool free_vec;
	uint8_t* vec=strings->get_feature_vector(num, vlen, free_vec);
	strings->free_feature_vector(vec, num, free_vec);
	return degree*vlen;
}

EFeatureType WDFeatures::get_feature_type() const
{
	return F_UNKNOWN;
}

EFeatureClass WDFeatures::get_feature_class() const
{
	return C_WD;
}

int32_t WDFeatures::get_num_vectors() const
{
	return num_strings;
}

float64_t WDFeatures::get_normalization_const()
{
	return normalization_const;
}

void WDFeatures::set_wd_weights(SGVector<float64_t> weights)
{
	ASSERT(weights.vlen==degree)

	for (int32_t i=0; i<degree; i++)
		wd_weights[i]=weights.vector[i];
}

