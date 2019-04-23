/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Vladislav Horbatiuk, Evgeniy Andreev
 */

#include <shogun/features/SNPFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/Alphabet.h>
#include <shogun/lib/memory.h>

using namespace shogun;

SNPFeatures::SNPFeatures()
{
	unstable(SOURCE_LOCATION);

	strings = NULL;

	string_length = 0;
	num_strings = 0;
	w_dim = 0;

	normalization_const = 0.0;

	m_str_min = NULL;
	m_str_maj = NULL;
}

SNPFeatures::SNPFeatures(std::shared_ptr<StringFeatures<uint8_t>> str) : DotFeatures(),
	m_str_min(NULL), m_str_maj(NULL)
{
	ASSERT(str)
	ASSERT(str->have_same_length())


	strings=str;
	string_length=str->get_max_vector_length();
	ASSERT((string_length & 1) == 0) // length divisible by 2
	w_dim=3*string_length/2;
	num_strings=str->get_num_vectors();
	auto alpha=str->get_alphabet();
	ASSERT(alpha->get_alphabet()==SNP)

	obtain_base_strings();
	set_normalization_const();

}

SNPFeatures::SNPFeatures(const SNPFeatures& orig)
	: DotFeatures(orig), strings(orig.strings),
	normalization_const(orig.normalization_const),
	m_str_min(NULL), m_str_maj(NULL)
{


	if (strings)
	{
		string_length=strings->get_max_vector_length();
		ASSERT((string_length & 1) == 0) // length divisible by 2
			w_dim=3*string_length;
		num_strings=strings->get_num_vectors();
	}
	else
	{
		string_length = 0;
		w_dim = 0;
		num_strings = 0;
	}

	obtain_base_strings();
}

SNPFeatures::~SNPFeatures()
{

}

int32_t SNPFeatures::get_dim_feature_space() const
{
	return w_dim;
}

int32_t SNPFeatures::get_nnz_features_for_vector(int32_t num) const
{
	return w_dim/3;
}

EFeatureType SNPFeatures::get_feature_type() const
{
	return F_UNKNOWN;
}

EFeatureClass SNPFeatures::get_feature_class() const
{
	return C_WD;
}

int32_t SNPFeatures::get_num_vectors() const
{
	return num_strings;
}

float64_t SNPFeatures::get_normalization_const()
{
	return normalization_const;
}

void SNPFeatures::set_minor_base_string(const char* str)
{
	m_str_min=(uint8_t*) get_strdup(str);
}

void SNPFeatures::set_major_base_string(const char* str)
{
	m_str_maj=(uint8_t*) get_strdup(str);
}

char* SNPFeatures::get_minor_base_string()
{
	return (char*) m_str_min;
}

char* SNPFeatures::get_major_base_string()
{
	return (char*) m_str_maj;
}

float64_t SNPFeatures::dot(int32_t idx_a, std::shared_ptr<DotFeatures> df, int32_t idx_b) const
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	auto sf=std::static_pointer_cast<SNPFeatures>(df);

	int32_t alen, blen;
	bool free_avec, free_bvec;

	uint8_t* avec = strings->get_feature_vector(idx_a, alen, free_avec);
	uint8_t* bvec = sf->strings->get_feature_vector(idx_b, blen, free_bvec);

	ASSERT(alen==blen)
	if (alen!=string_length)
		error("alen ({}) !=string_length ({})", alen, string_length);
	ASSERT(m_str_min)
	ASSERT(m_str_maj)

	float64_t total=0;

	for (int32_t i = 0; i<alen-1; i+=2)
	{
		int32_t sumaa=0;
		int32_t sumbb=0;
		int32_t sumab=0;

		uint8_t a1=avec[i];
		uint8_t a2=avec[i+1];
		uint8_t b1=bvec[i];
		uint8_t b2=bvec[i+1];

		if ((a1!=a2 || a1=='0' || a1=='0') && (b1!=b2 || b1=='0' || b2=='0'))
			sumab++;
		else if (a1==a2 && b1==b2)
		{
			if (a1!=b1)
				continue;

			if (a1==m_str_min[i])
				sumaa++;
			else if (a1==m_str_maj[i])
				sumbb++;
			else
			{
				error("The impossible happened i={} a1={} "
						"a2={} b1={} b2={} min={} maj={}", i, a1,a2, b1,b2, m_str_min[i], m_str_maj[i]);
			}

		}
		total+=sumaa+sumbb+sumab;
	}

	strings->free_feature_vector(avec, idx_a, free_avec);
	sf->strings->free_feature_vector(bvec, idx_b, free_bvec);
	return total;
}

float64_t
SNPFeatures::dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const
{
	require(
	    vec2.size() == w_dim, "Dimensions don't match, vec2_dim={}, w_dim={}",
	    vec2.size(), w_dim);

	float64_t sum=0;
	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	int32_t offs=0;

	for (int32_t i=0; i<len; i+=2)
	{
		int32_t dim=0;

		char a1=vec[i];
		char a2=vec[i+1];

		if (a1==a2 && a1!='0' && a2!='0')
		{
			if (a1==m_str_min[i])
				dim=1;
			else if (a1==m_str_maj[i])
				dim=2;
			else
			{
				error("The impossible happened i={} a1={} a2={} min={} maj={}",
						i, a1,a2, m_str_min[i], m_str_maj[i]);
			}
		}

		sum+=vec2[offs+dim];
		offs+=3;
	}
	strings->free_feature_vector(vec, vec_idx1, free_vec1);

	return sum/normalization_const;
}

void SNPFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val) const
{
	if (vec2_len != w_dim)
		error("Dimensions don't match, vec2_dim={}, w_dim={}", vec2_len, w_dim);

	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	int32_t offs=0;

	if (abs_val)
		alpha=Math::abs(alpha);

	for (int32_t i=0; i<len; i+=2)
	{
		int32_t dim=0;

		char a1=vec[i];
		char a2=vec[i+1];

		if (a1==a2 && a1!='0' && a2!='0')
		{
			if (a1==m_str_min[i])
				dim=1;
			else if (a1==m_str_maj[i])
				dim=2;
			else
			{
				error("The impossible happened i={} a1={} a2={} min={} maj={}",
						i, a1,a2, m_str_min[i], m_str_maj[i]);
			}
		}

		vec2[offs+dim]+=alpha;
		offs+=3;
	}
	strings->free_feature_vector(vec, vec_idx1, free_vec1);
}

void SNPFeatures::find_minor_major_strings(uint8_t* minor, uint8_t* major)
{
	for (int32_t i=0; i<num_strings; i++)
	{
		int32_t len;
		bool free_vec;
		uint8_t* vec = strings->get_feature_vector(i, len, free_vec);
		ASSERT(string_length==len)

		for (int32_t j=0; j<len; j++)
		{
			// skip sequencing errors
			if (vec[j]=='0')
				continue;

			if (minor[j]==0)
				minor[j]=vec[j];
            else if (major[j]==0 && vec[j]!=minor[j])
				major[j]=vec[j];
		}

		strings->free_feature_vector(vec, i, free_vec);
	}
}

void SNPFeatures::obtain_base_strings(std::shared_ptr<SNPFeatures> snp)
{
	SG_FREE(m_str_min);
	SG_FREE(m_str_maj);
	size_t tlen=(string_length+1)*sizeof(uint8_t);

	m_str_min=SG_CALLOC(uint8_t, tlen);
	m_str_maj=SG_CALLOC(uint8_t, tlen);

	find_minor_major_strings(m_str_min, m_str_maj);

	if (snp)
		snp->find_minor_major_strings(m_str_min, m_str_maj);

	for (int32_t j=0; j<string_length; j++)
	{
        // if only one symbol occurs use 0
		if (m_str_min[j]==0)
            m_str_min[j]='0';
		if (m_str_maj[j]==0)
            m_str_maj[j]='0';

		if (m_str_min[j]>m_str_maj[j])
			Math::swap(m_str_min[j], m_str_maj[j]);
	}
}

void SNPFeatures::set_normalization_const(float64_t n)
{
	if (n==0)
	{
		normalization_const=string_length;
		normalization_const = std::sqrt(normalization_const);
	}
	else
		normalization_const=n;

	SG_DEBUG("normalization_const:{}", normalization_const)
}

void* SNPFeatures::get_feature_iterator(int32_t vector_index)
{
	return NULL;
}

bool SNPFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	return false;
}

void SNPFeatures::free_feature_iterator(void* iterator)
{
}

std::shared_ptr<Features> SNPFeatures::duplicate() const
{
	return std::make_shared<SNPFeatures>(*this);
}

SGMatrix<float64_t> SNPFeatures::get_histogram(bool normalize)
{
	int32_t nsym=3;
	float64_t* h= SG_CALLOC(float64_t, size_t(nsym)*string_length/2);

	float64_t* h_normalizer=SG_MALLOC(float64_t, string_length/2);
	memset(h_normalizer, 0, string_length/2*sizeof(float64_t));
	int32_t num_str=get_num_vectors();
	for (int32_t i=0; i<num_str; i++)
	{
		int32_t len;
		bool free_vec;
		uint8_t* vec = strings->get_feature_vector(i, len, free_vec);

		for (int32_t j=0; j<len; j+=2)
		{
			int32_t dim=0;

			char a1=vec[j];
			char a2=vec[j+1];

			if (a1==a2 && a1!='0' && a2!='0')
			{
				if (a1==m_str_min[j])
					dim=1;
				else if (a1==m_str_maj[j])
					dim=2;
				else
				{
					error("The impossible happened j={} a1={} a2={} min={} maj={}",
							j, a1,a2, m_str_min[j], m_str_maj[j]);
				}
			}

			h[int64_t(j/2)*nsym+dim]++;
			h_normalizer[j/2]++;
		}

		strings->free_feature_vector(vec, i, free_vec);
	}

	if (normalize)
	{
		for (int32_t i=0; i<string_length/2; i++)
		{
			for (int32_t j=0; j<nsym; j++)
			{
				if (h_normalizer && h_normalizer[i])
					h[int64_t(i)*nsym+j]/=h_normalizer[i];
			}
		}
	}
	SG_FREE(h_normalizer);

	return SGMatrix<float64_t>(h, nsym, string_length/2);
}

SGMatrix<float64_t> SNPFeatures::get_2x3_table(std::shared_ptr<SNPFeatures> pos, std::shared_ptr<SNPFeatures> neg)
{

	ASSERT(pos->strings->get_max_vector_length() == neg->strings->get_max_vector_length())
	int32_t len=pos->strings->get_max_vector_length();

	float64_t* table=SG_MALLOC(float64_t, 3*2*len/2);

	SGMatrix<float64_t> p_hist=pos->get_histogram(false);
	SGMatrix<float64_t> n_hist=neg->get_histogram(false);


	for (int32_t i=0; i<3*len/2; i++)
	{
		table[2*i]=p_hist.matrix[i];
		table[2*i+1]=n_hist.matrix[i];
	}
	return SGMatrix<float64_t>(table, 2,3*len/2);
}
