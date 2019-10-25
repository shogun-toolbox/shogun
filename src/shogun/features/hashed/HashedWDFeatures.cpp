/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#include <shogun/features/hashed/HashedWDFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

HashedWDFeatures::HashedWDFeatures() :DotFeatures()
{
	unstable(SOURCE_LOCATION);

	strings = NULL;

	degree = 0;
	start_degree = 0;
	from_degree = 0;
	string_length = 0;
	num_strings = 0;
	alphabet_size = 0;
	w_dim = 0;
	partial_w_dim = 0;
	wd_weights = NULL;
	mask = 0;
	m_hash_bits = 0;

	normalization_const = 0.0;
}

HashedWDFeatures::HashedWDFeatures(std::shared_ptr<StringFeatures<uint8_t>> str,
		int32_t start_order, int32_t order, int32_t from_order,
		int32_t hash_bits) : DotFeatures()
{
	ASSERT(start_order>=0)
	ASSERT(start_order<order)
	ASSERT(order<=from_order)
	ASSERT(hash_bits>0)
	ASSERT(str)
	ASSERT(str->have_same_length())


	strings=str;
	string_length=str->get_max_vector_length();
	num_strings=str->get_num_vectors();
	auto alpha=str->get_alphabet();
	alphabet_size=alpha->get_num_symbols();

	degree=order;
	start_degree=start_order;
	from_degree=from_order;
	m_hash_bits=hash_bits;
	set_wd_weights();
	set_normalization_const();
}

HashedWDFeatures::HashedWDFeatures(const HashedWDFeatures& orig)
	: DotFeatures(orig), strings(orig.strings),
	degree(orig.degree), start_degree(orig.start_degree),
	from_degree(orig.from_degree), m_hash_bits(orig.m_hash_bits),
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

	if (degree>0)
		set_wd_weights();
}

HashedWDFeatures::~HashedWDFeatures()
{

	SG_FREE(wd_weights);
}

float64_t HashedWDFeatures::dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	auto wdf = std::static_pointer_cast<HashedWDFeatures>(df);

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
				break;
			if (j>=start_degree)
				sum += wd_weights[j]*wd_weights[j];
		}
	}
	strings->free_feature_vector(vec1, vec_idx1, free_vec1);
	wdf->strings->free_feature_vector(vec2, vec_idx2, free_vec2);
	return sum/Math::sq(normalization_const);
}

float64_t
HashedWDFeatures::dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const
{
	require(
	    vec2.size() == w_dim, "Dimensions don't match, vec2_dim={}, w_dim={}",
	    vec2.size(), w_dim);

	float64_t sum=0;
	int32_t lim=Math::min(degree, string_length);
	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	uint32_t* val=SG_MALLOC(uint32_t, len);

	uint32_t offs=0;

	if (start_degree>0)
	{
		// compute hash for strings of length start_degree-1
		for (int32_t i=0; i+start_degree < len; i++)
			val[i]=Hash::MurmurHash3(&vec[i], start_degree, 0xDEADBEAF);
	}
	else
		SGVector<uint32_t>::fill_vector(val, len, 0xDEADBEAF);

	for (int32_t k=start_degree; k<lim; k++)
	{
		float64_t wd = wd_weights[k];

		uint32_t o=offs;
		uint32_t carry = 0;
		uint32_t chunk = 0;

		for (int32_t i=0; i+k < len; i++)
		{
			chunk++;
			Hash::IncrementalMurmurHash3(&(val[i]), &carry, &(vec[i+k]), 1);
			uint32_t h =
					Hash::FinalizeIncrementalMurmurHash3(val[i], carry, chunk);
#ifdef DEBUG_HASHEDWD
			io::print("vec[i]={}, k={}, offs={} o={}\n", vec[i], k,offs, o);
#endif
			sum+=vec2[o+(h & mask)]*wd;
			val[i] = h;
			o+=partial_w_dim;
		}
		val[len-k-1] =
				Hash::FinalizeIncrementalMurmurHash3(val[len-k-1], carry, chunk);
		offs+=partial_w_dim*len;
	}
	SG_FREE(val);
	strings->free_feature_vector(vec, vec_idx1, free_vec1);

	return sum/normalization_const;
}

void HashedWDFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val) const
{
	if (vec2_len != w_dim)
		error("Dimensions don't match, vec2_dim={}, w_dim={}", vec2_len, w_dim);

	int32_t lim=Math::min(degree, string_length);
	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	uint32_t* val=SG_MALLOC(uint32_t, len);

	uint32_t offs=0;

	if (start_degree>0)
	{
		// compute hash for strings of length start_degree-1
		for (int32_t i=0; i+start_degree < len; i++)
			val[i]=Hash::MurmurHash3(&vec[i], start_degree, 0xDEADBEAF);
	}
	else
		SGVector<uint32_t>::fill_vector(val, len, 0xDEADBEAF);

	for (int32_t k=start_degree; k<lim; k++)
	{
		float64_t wd = alpha*wd_weights[k]/normalization_const;

		if (abs_val)
			wd=Math::abs(wd);

		uint32_t o=offs;
		uint32_t carry = 0;
		uint32_t chunk = 0;

		for (int32_t i=0; i+k < len; i++)
		{
			chunk++;
			Hash::IncrementalMurmurHash3(&(val[i]), &carry, &(vec[i+k]), 1);
			uint32_t h = Hash::FinalizeIncrementalMurmurHash3(val[i], carry, chunk);

#ifdef DEBUG_HASHEDWD
			io::print("offs={} o={} h={} \n", offs, o, h);
			io::print("vec[i]={}, k={}, offs={} o={}\n", vec[i], k,offs, o);
#endif
			vec2[o+(h & mask)]+=wd;
			val[i] = h;
			o+=partial_w_dim;
		}
		val[len-k-1] =
				Hash::FinalizeIncrementalMurmurHash3(val[len-k-1], carry, chunk);

		offs+=partial_w_dim*len;
	}

	SG_FREE(val);
	strings->free_feature_vector(vec, vec_idx1, free_vec1);
}

void HashedWDFeatures::set_wd_weights()
{
	ASSERT(degree>0)

	mask=(uint32_t) (((uint64_t) 1)<<m_hash_bits)-1;
	partial_w_dim=1<<m_hash_bits;
	w_dim=partial_w_dim*string_length*(degree-start_degree);

	wd_weights=SG_MALLOC(float64_t, degree);

	for (int32_t i=0; i<degree; i++)
		wd_weights[i]=sqrt(2.0*(from_degree-i)/(from_degree*(from_degree+1)));

	SG_DEBUG("created HashedWDFeatures with d={} ({}), alphabetsize={}, "
			"dim={} partial_dim={} num={}, len={}",
			degree, from_degree, alphabet_size,
			w_dim, partial_w_dim, num_strings, string_length);
}


void HashedWDFeatures::set_normalization_const(float64_t n)
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

std::shared_ptr<Features> HashedWDFeatures::duplicate() const
{
	return std::make_shared<HashedWDFeatures>(*this);
}


int32_t HashedWDFeatures::get_nnz_features_for_vector(int32_t num) const
{
	int32_t vlen=-1;
	bool free_vec;
	uint8_t* vec=strings->get_feature_vector(num, vlen, free_vec);
	strings->free_feature_vector(vec, num, free_vec);
	return degree*vlen;
}

void* HashedWDFeatures::get_feature_iterator(int32_t vector_index)
{
	not_implemented(SOURCE_LOCATION);
	return NULL;
}

bool HashedWDFeatures::get_next_feature(int32_t& index, float64_t& value,
		void* iterator)
{
	not_implemented(SOURCE_LOCATION);
	return false;
}

void HashedWDFeatures::free_feature_iterator(void* iterator)
{
	not_implemented(SOURCE_LOCATION);
}
