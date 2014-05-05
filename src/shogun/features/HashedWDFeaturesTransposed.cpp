/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/features/HashedWDFeaturesTransposed.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/base/Parallel.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct HASHEDWD_THREAD_PARAM
{
	CHashedWDFeaturesTransposed* hf;
	int32_t* sub_index;
	float64_t* output;
	int32_t start;
	int32_t stop;
	float64_t* alphas;
	float64_t* vec;
	float64_t bias;
	bool progress;
	uint32_t* index;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

CHashedWDFeaturesTransposed::CHashedWDFeaturesTransposed()
	:CDotFeatures()
{
	SG_UNSTABLE(
		"CHashedWDFeaturesTransposed::CHashedWDFeaturesTransposed()",
		"\n");

	strings = NULL;
	transposed_strings = NULL;

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

CHashedWDFeaturesTransposed::CHashedWDFeaturesTransposed(CStringFeatures<uint8_t>* str,
		int32_t start_order, int32_t order, int32_t from_order,
		int32_t hash_bits) : CDotFeatures()
{
	ASSERT(start_order>=0)
	ASSERT(start_order<order)
	ASSERT(order<=from_order)
	ASSERT(hash_bits>0)
	ASSERT(str)
	ASSERT(str->have_same_length())
	SG_REF(str);

	strings=str;
	int32_t transposed_num_feat=0;
	int32_t transposed_num_vec=0;
	transposed_strings=str->get_transposed(transposed_num_feat, transposed_num_vec);

	string_length=str->get_max_vector_length();
	num_strings=str->get_num_vectors();
	ASSERT(transposed_num_feat==num_strings)
	ASSERT(transposed_num_vec==string_length)

	CAlphabet* alpha=str->get_alphabet();
	alphabet_size=alpha->get_num_symbols();
	SG_UNREF(alpha);

	degree=order;
	start_degree=start_order;
	if (start_degree!=0)
		SG_NOTIMPLEMENTED
	from_degree=from_order;
	m_hash_bits=hash_bits;
	set_wd_weights();
	set_normalization_const();
}

CHashedWDFeaturesTransposed::~CHashedWDFeaturesTransposed()
{
	for (int32_t i=0; i<string_length; i++)
		SG_FREE(transposed_strings[i].string);
	SG_FREE(transposed_strings);

	SG_UNREF(strings);
	SG_FREE(wd_weights);
}

float64_t CHashedWDFeaturesTransposed::dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	CHashedWDFeaturesTransposed* wdf = (CHashedWDFeaturesTransposed*) df;

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
	return sum/CMath::sq(normalization_const);
}

float64_t CHashedWDFeaturesTransposed::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	if (vec2_len != w_dim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, w_dim=%d\n", vec2_len, w_dim)

	float64_t sum=0;
	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	uint32_t* val=SG_MALLOC(uint32_t, len);

	uint32_t offs=0;

	SGVector<uint32_t>::fill_vector(val, len, 0xDEADBEAF);

	for (int32_t i=0; i < len; i++)
	{
		uint32_t o=offs;
		uint32_t carry = 0;
		uint32_t chunk = 0;

		for (int32_t k=0; k<degree && i+k<len; k++)
		{
			const float64_t wd = wd_weights[k];
			chunk++;
			CHash::IncrementalMurmurHash3(&(val[i]), &carry, &(vec[i+k]), 1);
			uint32_t h =
					CHash::FinalizeIncrementalMurmurHash3(val[i], carry, chunk);
#ifdef DEBUG_HASHEDWD
			SG_PRINT("vec[i]=%d, k=%d, offs=%d o=%d h=%d \n", vec[i], k,offs, o, h)
#endif
			sum+=vec2[o+(h & mask)]*wd;
			o+=partial_w_dim;
		}
		val[i] = CHash::FinalizeIncrementalMurmurHash3(val[i], carry, chunk);
		offs+=partial_w_dim*degree;
	}
	SG_FREE(val);
	strings->free_feature_vector(vec, vec_idx1, free_vec1);

	return sum/normalization_const;
}

void CHashedWDFeaturesTransposed::dense_dot_range(float64_t* output, int32_t start, int32_t stop, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b)
{
	ASSERT(output)
	// write access is internally between output[start..stop] so the following
	// line is necessary to write to output[0...(stop-start-1)]
	output-=start;
	ASSERT(start>=0)
	ASSERT(start<stop)
	ASSERT(stop<=get_num_vectors())
	uint32_t* index=SG_MALLOC(uint32_t, stop);

	int32_t num_vectors=stop-start;
	ASSERT(num_vectors>0)

	int32_t num_threads=parallel->get_num_threads();
	ASSERT(num_threads>0)

	CSignal::clear_cancel();

	if (dim != w_dim)
		SG_ERROR("Dimensions don't match, vec_len=%d, w_dim=%d\n", dim, w_dim)

#ifdef HAVE_PTHREAD
	if (num_threads < 2)
	{
#endif
		HASHEDWD_THREAD_PARAM params;
		params.hf=this;
		params.sub_index=NULL;
		params.output=output;
		params.start=start;
		params.stop=stop;
		params.alphas=alphas;
		params.vec=vec;
		params.bias=b;
		params.progress=false; //true;
		params.index=index;
		dense_dot_range_helper((void*) &params);
#ifdef HAVE_PTHREAD
	}
	else
	{
		pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
		HASHEDWD_THREAD_PARAM* params = SG_MALLOC(HASHEDWD_THREAD_PARAM, num_threads);
		int32_t step= num_vectors/num_threads;

		int32_t t;

		for (t=0; t<num_threads-1; t++)
		{
			params[t].hf = this;
			params[t].sub_index=NULL;
			params[t].output = output;
			params[t].start = start+t*step;
			params[t].stop = start+(t+1)*step;
			params[t].alphas=alphas;
			params[t].vec=vec;
			params[t].bias=b;
			params[t].progress = false;
			params[t].index=index;
			pthread_create(&threads[t], NULL,
					CHashedWDFeaturesTransposed::dense_dot_range_helper, (void*)&params[t]);
		}

		params[t].hf = this;
		params[t].sub_index=NULL;
		params[t].output = output;
		params[t].start = start+t*step;
		params[t].stop = stop;
		params[t].alphas=alphas;
		params[t].vec=vec;
		params[t].bias=b;
		params[t].progress = false; //true;
		params[t].index=index;
		CHashedWDFeaturesTransposed::dense_dot_range_helper((void*) &params[t]);

		for (t=0; t<num_threads-1; t++)
			pthread_join(threads[t], NULL);

		SG_FREE(params);
		SG_FREE(threads);
	}
#endif
	SG_FREE(index);

#ifndef WIN32
		if ( CSignal::cancel_computations() )
			SG_INFO("prematurely stopped.           \n")
#endif
}

void CHashedWDFeaturesTransposed::dense_dot_range_subset(int32_t* sub_index, int num, float64_t* output, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b)
{
	ASSERT(sub_index)
	ASSERT(output)

	uint32_t* index=SG_MALLOC(uint32_t, num);

	int32_t num_threads=parallel->get_num_threads();
	ASSERT(num_threads>0)

	CSignal::clear_cancel();

	if (dim != w_dim)
		SG_ERROR("Dimensions don't match, vec_len=%d, w_dim=%d\n", dim, w_dim)

#ifdef HAVE_PTHREAD
	if (num_threads < 2)
	{
#endif
		HASHEDWD_THREAD_PARAM params;
		params.hf=this;
		params.sub_index=sub_index;
		params.output=output;
		params.start=0;
		params.stop=num;
		params.alphas=alphas;
		params.vec=vec;
		params.bias=b;
		params.progress=false; //true;
		params.index=index;
		dense_dot_range_helper((void*) &params);
#ifdef HAVE_PTHREAD
	}
	else
	{
		pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
		HASHEDWD_THREAD_PARAM* params = SG_MALLOC(HASHEDWD_THREAD_PARAM, num_threads);
		int32_t step= num/num_threads;

		int32_t t;

		for (t=0; t<num_threads-1; t++)
		{
			params[t].hf = this;
			params[t].sub_index=sub_index;
			params[t].output = output;
			params[t].start = t*step;
			params[t].stop = (t+1)*step;
			params[t].alphas=alphas;
			params[t].vec=vec;
			params[t].bias=b;
			params[t].progress = false;
			params[t].index=index;
			pthread_create(&threads[t], NULL,
					CHashedWDFeaturesTransposed::dense_dot_range_helper, (void*)&params[t]);
		}

		params[t].hf = this;
		params[t].sub_index=sub_index;
		params[t].output = output;
		params[t].start = t*step;
		params[t].stop = num;
		params[t].alphas=alphas;
		params[t].vec=vec;
		params[t].bias=b;
		params[t].progress = false; //true;
		params[t].index=index;
		CHashedWDFeaturesTransposed::dense_dot_range_helper((void*) &params[t]);

		for (t=0; t<num_threads-1; t++)
			pthread_join(threads[t], NULL);

		SG_FREE(params);
		SG_FREE(threads);
		SG_FREE(index);
	}
#endif

#ifndef WIN32
		if ( CSignal::cancel_computations() )
			SG_INFO("prematurely stopped.           \n")
#endif
}

void* CHashedWDFeaturesTransposed::dense_dot_range_helper(void* p)
{
	HASHEDWD_THREAD_PARAM* par=(HASHEDWD_THREAD_PARAM*) p;
	CHashedWDFeaturesTransposed* hf=par->hf;
	int32_t* sub_index=par->sub_index;
	float64_t* output=par->output;
	int32_t start=par->start;
	int32_t stop=par->stop;
	float64_t* alphas=par->alphas;
	float64_t* vec=par->vec;
	float64_t bias=par->bias;
	bool progress=par->progress;
	uint32_t* index=par->index;
	int32_t string_length=hf->string_length;
	int32_t degree=hf->degree;
	float64_t* wd_weights=hf->wd_weights;
	SGString<uint8_t>* transposed_strings=hf->transposed_strings;
	uint32_t mask=hf->mask;
	int32_t partial_w_dim=hf->partial_w_dim;
	float64_t normalization_const=hf->normalization_const;

	if (sub_index)
	{
		for (int32_t j=start; j<stop; j++)
			output[j]=0.0;

		uint32_t offs=0;
		for (int32_t i=0; i<string_length; i++)
		{
			uint32_t o=offs;
			for (int32_t k=0; k<degree && i+k<string_length; k++)
			{
				const float64_t wd = wd_weights[k];
				uint8_t* dim=transposed_strings[i+k].string;
				uint32_t carry = 0;
				uint32_t chunk = 0;
				for (int32_t j=start; j<stop; j++)
				{
					uint8_t bval=dim[sub_index[j]];
					if (k==0)
						index[j] = 0xDEADBEAF;

					CHash::IncrementalMurmurHash3(&index[j], &carry, &bval, 1);

					chunk++;
					uint32_t h =
							CHash::FinalizeIncrementalMurmurHash3(
									index[j], carry, chunk);

					output[j]+=vec[o + (h & mask)]*wd;
					index[j] = h;
				}

				index[stop-1] =
						CHash::FinalizeIncrementalMurmurHash3(
								index[stop-1], carry, chunk);

				o+=partial_w_dim;
			}
			offs+=partial_w_dim*degree;

			if (progress)
				hf->io->progress(i, 0,string_length, i);
		}

		for (int32_t j=start; j<stop; j++)
		{
			if (alphas)
				output[j]=output[j]*alphas[sub_index[j]]/normalization_const+bias;
			else
				output[j]=output[j]/normalization_const+bias;
		}
	}
	else
	{
		SGVector<float64_t>::fill_vector(&output[start], stop-start, 0.0);

		uint32_t offs=0;
		for (int32_t i=0; i<string_length; i++)
		{
			uint32_t o=offs;
			for (int32_t k=0; k<degree && i+k<string_length; k++)
			{
				const float64_t wd = wd_weights[k];
				uint8_t* dim=transposed_strings[i+k].string;
				uint32_t carry = 0;
				uint32_t chunk = 0;

				for (int32_t j=start; j<stop; j++)
				{
					uint8_t bval=dim[sub_index[j]];
					if (k==0)
						index[j] = 0xDEADBEAF;

					CHash::IncrementalMurmurHash3(&index[j], &carry, &bval, 1);

					chunk++;
					uint32_t h =
							CHash::FinalizeIncrementalMurmurHash3(
									index[j], carry, chunk);

					index[j] = h;
					output[j]+=vec[o + (h & mask)]*wd;
				}

				index[stop-1] = CHash::FinalizeIncrementalMurmurHash3(
						index[stop-1], carry, chunk);

				o+=partial_w_dim;
			}
			offs+=partial_w_dim*degree;

			if (progress)
				hf->io->progress(i, 0,string_length, i);
		}

		for (int32_t j=start; j<stop; j++)
		{
			if (alphas)
				output[j]=output[j]*alphas[j]/normalization_const+bias;
			else
				output[j]=output[j]/normalization_const+bias;
		}
	}

	return NULL;
}

void CHashedWDFeaturesTransposed::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	if (vec2_len != w_dim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, w_dim=%d\n", vec2_len, w_dim)

	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	uint32_t* val=SG_MALLOC(uint32_t, len);

	uint32_t offs=0;
	float64_t factor=alpha/normalization_const;
	if (abs_val)
		factor=CMath::abs(factor);

	SGVector<uint32_t>::fill_vector(val, len, 0xDEADBEAF);

	for (int32_t i=0; i<len; i++)
	{
		uint32_t o=offs;
		uint32_t carry = 0;
		uint32_t chunk = 0;

		for (int32_t k=0; k<degree && i+k<len; k++)
		{
			float64_t wd = wd_weights[k]*factor;
			chunk++;
			CHash::IncrementalMurmurHash3(&(val[i]), &carry, &(vec[i+k]), 1);
			uint32_t h =
					CHash::FinalizeIncrementalMurmurHash3(val[i], carry, chunk);
#ifdef DEBUG_HASHEDWD
			SG_PRINT("offs=%d o=%d h=%d \n", offs, o, h)
			SG_PRINT("vec[i]=%d, k=%d, offs=%d o=%d\n", vec[i], k,offs, o)
#endif
			vec2[o+(h & mask)]+=wd;
			val[i] = h;
			o+=partial_w_dim;
		}

		val[i] = CHash::FinalizeIncrementalMurmurHash3(val[i], carry, chunk);
		offs+=partial_w_dim*degree;
	}

	SG_FREE(val);
	strings->free_feature_vector(vec, vec_idx1, free_vec1);
}

void CHashedWDFeaturesTransposed::set_wd_weights()
{
	ASSERT(degree>0)

	mask=(uint32_t) (((uint64_t) 1)<<m_hash_bits)-1;
	partial_w_dim=1<<m_hash_bits;
	w_dim=partial_w_dim*string_length*(degree-start_degree);

	wd_weights=SG_MALLOC(float64_t, degree);

	for (int32_t i=0; i<degree; i++)
		wd_weights[i]=sqrt(2.0*(from_degree-i)/(from_degree*(from_degree+1)));

	SG_DEBUG("created HashedWDFeaturesTransposed with d=%d (%d), alphabetsize=%d, "
			"dim=%d partial_dim=%d num=%d, len=%d\n",
			degree, from_degree, alphabet_size,
			w_dim, partial_w_dim, num_strings, string_length);
}


void CHashedWDFeaturesTransposed::set_normalization_const(float64_t n)
{
	if (n==0)
	{
		normalization_const=0;
		for (int32_t i=0; i<degree; i++)
			normalization_const+=(string_length-i)*wd_weights[i]*wd_weights[i];

		normalization_const=CMath::sqrt(normalization_const);
	}
	else
		normalization_const=n;

	SG_DEBUG("normalization_const:%f\n", normalization_const)
}

CFeatures* CHashedWDFeaturesTransposed::duplicate() const
{
	SG_NOTIMPLEMENTED
	// return new CHashedWDFeaturesTransposed(*this);
	return NULL;
}

void* CHashedWDFeaturesTransposed::get_feature_iterator(int32_t vector_index)
{
	SG_NOTIMPLEMENTED
	return NULL;
}

bool CHashedWDFeaturesTransposed::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	SG_NOTIMPLEMENTED
	return false;
}

void CHashedWDFeaturesTransposed::free_feature_iterator(void* iterator)
{
	SG_NOTIMPLEMENTED
}
