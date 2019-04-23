/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Weijie Lin, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/string/WeightedCommWordStringKernel.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

WeightedCommWordStringKernel::WeightedCommWordStringKernel()
  : CommWordStringKernel(0, false)
{
	init();
}

WeightedCommWordStringKernel::WeightedCommWordStringKernel(
	int32_t size, bool us)
: CommWordStringKernel(size, us)
{
	ASSERT(us==false)
	init();
}

WeightedCommWordStringKernel::WeightedCommWordStringKernel(
	std::shared_ptr<StringFeatures<uint16_t>> l, std::shared_ptr<StringFeatures<uint16_t>> r, bool us,
	int32_t size)
: CommWordStringKernel(size, us)
{
	ASSERT(us==false)
	init();

	init(l,r);
}

WeightedCommWordStringKernel::~WeightedCommWordStringKernel()
{
	SG_FREE(weights);
}

bool WeightedCommWordStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	auto sf_l = std::static_pointer_cast<StringFeatures<uint16_t>>(l);
	auto sf_r = std::static_pointer_cast<StringFeatures<uint16_t>>(r);
	ASSERT(sf_l->get_order() ==	sf_r->get_order());
	degree=sf_l->get_order();
	set_wd_weights();

	CommWordStringKernel::init(l,r);
	return init_normalizer();
}

void WeightedCommWordStringKernel::cleanup()
{
	SG_FREE(weights);
	weights=NULL;

	CommWordStringKernel::cleanup();
}

bool WeightedCommWordStringKernel::set_wd_weights()
{
	SG_FREE(weights);
	weights=SG_MALLOC(float64_t, degree);

	int32_t i;
	float64_t sum=0;
	for (i=0; i<degree; i++)
	{
		weights[i]=degree-i;
		sum+=weights[i];
	}
	for (i=0; i<degree; i++)
		weights[i] = std::sqrt(weights[i] / sum);

	return weights!=NULL;
}

bool WeightedCommWordStringKernel::set_weights(SGVector<float64_t> w)
{
	ASSERT(w.vlen==degree)

	SG_FREE(weights);
	weights = w.vector;
	for (int32_t i=0; i<degree; i++)
		weights[i] = std::sqrt(weights[i]);
	return true;
}

float64_t WeightedCommWordStringKernel::compute_helper(
	int32_t idx_a, int32_t idx_b, bool do_sort)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	auto l = std::static_pointer_cast<StringFeatures<uint16_t>>(lhs);
	auto r = std::static_pointer_cast<StringFeatures<uint16_t>>(rhs);

	uint16_t* av=l->get_feature_vector(idx_a, alen, free_avec);
	uint16_t* bv=r->get_feature_vector(idx_b, blen, free_bvec);

	uint16_t* avec=av;
	uint16_t* bvec=bv;

	if (do_sort)
	{
		if (alen>0)
		{
			avec=SG_MALLOC(uint16_t, alen);
			sg_memcpy(avec, av, sizeof(uint16_t)*alen);
			Math::radix_sort(avec, alen);
		}
		else
			avec=NULL;

		if (blen>0)
		{
			bvec=SG_MALLOC(uint16_t, blen);
			sg_memcpy(bvec, bv, sizeof(uint16_t)*blen);
			Math::radix_sort(bvec, blen);
		}
		else
			bvec=NULL;
	}

	float64_t result=0;
	uint8_t mask=0;

	for (int32_t d=0; d<degree; d++)
	{
		mask = mask | (1 << (degree-d-1));
		uint16_t masked=std::static_pointer_cast<StringFeatures<uint16_t>>(lhs)->get_masked_symbols(0xffff, mask);

		int32_t left_idx=0;
		int32_t right_idx=0;
		float64_t weight=weights[d]*weights[d];

		while (left_idx < alen && right_idx < blen)
		{
			uint16_t lsym=avec[left_idx] & masked;
			uint16_t rsym=bvec[right_idx] & masked;

			if (lsym == rsym)
			{
				int32_t old_left_idx=left_idx;
				int32_t old_right_idx=right_idx;

				while (left_idx<alen && (avec[left_idx] & masked) ==lsym)
					left_idx++;

				while (right_idx<blen && (bvec[right_idx] & masked) ==lsym)
					right_idx++;

				result+=weight*(left_idx-old_left_idx)*(right_idx-old_right_idx);
			}
			else if (lsym<rsym)
				left_idx++;
			else
				right_idx++;
		}
	}

	if (do_sort)
	{
		SG_FREE(avec);
		SG_FREE(bvec);
	}

	l->free_feature_vector(av, idx_a, free_avec);
	r->free_feature_vector(bv, idx_b, free_bvec);

	return result;
}

void WeightedCommWordStringKernel::add_to_normal(
	int32_t vec_idx, float64_t weight)
{
	int32_t len=-1;
	bool free_vec;
	auto s=std::static_pointer_cast<StringFeatures<uint16_t>>(lhs);
	uint16_t* vec=s->get_feature_vector(vec_idx, len, free_vec);

	if (len>0)
	{
		for (int32_t j=0; j<len; j++)
		{
			uint8_t mask=0;
			int32_t offs=0;
			for (int32_t d=0; d<degree; d++)
			{
				mask = mask | (1 << (degree-d-1));
				int32_t idx=s->get_masked_symbols(vec[j], mask);
				idx=s->shift_symbol(idx, degree-d-1);
				dictionary_weights[offs + idx] += normalizer->normalize_lhs(weight*weights[d], vec_idx);
				offs+=s->shift_offset(1,d+1);
			}
		}

		set_is_initialized(true);
	}

	s->free_feature_vector(vec, vec_idx, free_vec);
}

void WeightedCommWordStringKernel::merge_normal()
{
	ASSERT(get_is_initialized())
	ASSERT(use_sign==false)

	auto s=std::static_pointer_cast<StringFeatures<uint16_t>>(rhs);
	uint32_t num_symbols=(uint32_t) s->get_num_symbols();
	int32_t dic_size=1<<(sizeof(uint16_t)*8);
	float64_t* dic=SG_MALLOC(float64_t, dic_size);
	memset(dic, 0, sizeof(float64_t)*dic_size);

	for (uint32_t sym=0; sym<num_symbols; sym++)
	{
		float64_t result=0;
		uint8_t mask=0;
		int32_t offs=0;
		for (int32_t d=0; d<degree; d++)
		{
			mask = mask | (1 << (degree-d-1));
			int32_t idx=s->get_masked_symbols(sym, mask);
			idx=s->shift_symbol(idx, degree-d-1);
			result += dictionary_weights[offs + idx];
			offs+=s->shift_offset(1,d+1);
		}
		dic[sym]=result;
	}

	init_dictionary(1<<(sizeof(uint16_t)*8));
	sg_memcpy(dictionary_weights, dic, sizeof(float64_t)*dic_size);
	SG_FREE(dic);
}

float64_t WeightedCommWordStringKernel::compute_optimized(int32_t i)
{
	if (!get_is_initialized())
		SG_ERROR("CCommWordStringKernel optimization not initialized\n")

	ASSERT(use_sign==false)

	float64_t result=0;
	bool free_vec;
	int32_t len=-1;
	auto s=std::static_pointer_cast<StringFeatures<uint16_t>>(rhs);
	uint16_t* vec=s->get_feature_vector(i, len, free_vec);

	if (vec && len>0)
	{
		for (int32_t j=0; j<len; j++)
		{
			uint8_t mask=0;
			int32_t offs=0;
			for (int32_t d=0; d<degree; d++)
			{
				mask = mask | (1 << (degree-d-1));
				int32_t idx=s->get_masked_symbols(vec[j], mask);
				idx=s->shift_symbol(idx, degree-d-1);
				result += dictionary_weights[offs + idx]*weights[d];
				offs+=s->shift_offset(1,d+1);
			}
		}

		result=normalizer->normalize_rhs(result, i);
	}
	s->free_feature_vector(vec, i, free_vec);
	return result;
}

float64_t* WeightedCommWordStringKernel::compute_scoring(
	int32_t max_degree, int32_t& num_feat, int32_t& num_sym, float64_t* target,
	int32_t num_suppvec, int32_t* IDX, float64_t* alphas, bool do_init)
{
	if (do_init)
		CommWordStringKernel::init_optimization(num_suppvec, IDX, alphas);

	int32_t dic_size=1<<(sizeof(uint16_t)*9);
	float64_t* dic=SG_MALLOC(float64_t, dic_size);
	sg_memcpy(dic, dictionary_weights, sizeof(float64_t)*dic_size);

	merge_normal();
	float64_t* result=CommWordStringKernel::compute_scoring(max_degree, num_feat,
			num_sym, target, num_suppvec, IDX, alphas, false);

	init_dictionary(1<<(sizeof(uint16_t)*9));
	sg_memcpy(dictionary_weights,dic,  sizeof(float64_t)*dic_size);
	SG_FREE(dic);

	return result;
}

void WeightedCommWordStringKernel::init()
{
	degree=0;
	weights=NULL;

	init_dictionary(1<<(sizeof(uint16_t)*9));

	/*m_parameters->add_vector(&weights, &degree, "weights",
			"weights for each of the subkernels of degree 1...d");*/
	watch_param("weights", &weights, &degree);
}
