/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Soeren Sonnenburg, Sergey Lisitsyn,
 *          Evangelos Anagnostopoulos, Leon Kuchenbecker
 */

#include <shogun/base/progress.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/string/CommUlongStringKernel.h>
#include <shogun/lib/common.h>

#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>

using namespace shogun;

CommUlongStringKernel::CommUlongStringKernel() : StringKernel<uint64_t>()
{
	init_params();
}

CommUlongStringKernel::CommUlongStringKernel(bool us, int32_t size)
: StringKernel<uint64_t>(size)
{
	init_params();

	use_sign = us;

	properties |= KP_LINADD;
	clear_normal();

	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());
}

CommUlongStringKernel::CommUlongStringKernel(
	std::shared_ptr<StringFeatures<uint64_t>> l, std::shared_ptr<StringFeatures<uint64_t>> r, bool us,
	int32_t size)
: StringKernel<uint64_t>(size)
{
	init_params();
	use_sign=us;
	properties |= KP_LINADD;
	clear_normal();
	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());
	init(l,r);
}

void CommUlongStringKernel::init_params()
{
	use_sign = false;
	SG_ADD(&use_sign, "use_sign", "Whether or not to use sign.");
}

CommUlongStringKernel::~CommUlongStringKernel()
{
	cleanup();
}

void CommUlongStringKernel::remove_lhs()
{
	delete_optimization();

#ifdef SVMLIGHT
	if (lhs)
		cache_reset();
#endif

	lhs = NULL ;
	rhs = NULL ;
}

void CommUlongStringKernel::remove_rhs()
{
#ifdef SVMLIGHT
	if (rhs)
		cache_reset();
#endif

	rhs = lhs;
}

bool CommUlongStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	StringKernel<uint64_t>::init(l,r);
	return init_normalizer();
}

void CommUlongStringKernel::cleanup()
{
	delete_optimization();
	clear_normal();
	Kernel::cleanup();
}

float64_t CommUlongStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;
	uint64_t* avec=(std::static_pointer_cast<StringFeatures<uint64_t>>(lhs))->get_feature_vector(idx_a, alen, free_avec);
	uint64_t* bvec=(std::static_pointer_cast<StringFeatures<uint64_t>>(rhs))->get_feature_vector(idx_b, blen, free_bvec);

	float64_t result=0;

	int32_t left_idx=0;
	int32_t right_idx=0;

	if (use_sign)
	{
		while (left_idx < alen && right_idx < blen)
		{
			if (avec[left_idx]==bvec[right_idx])
			{
				uint64_t sym=avec[left_idx];

				while (left_idx< alen && avec[left_idx]==sym)
					left_idx++;

				while (right_idx< blen && bvec[right_idx]==sym)
					right_idx++;

				result++;
			}
			else if (avec[left_idx]<bvec[right_idx])
				left_idx++;
			else
				right_idx++;
		}
	}
	else
	{
		while (left_idx < alen && right_idx < blen)
		{
			if (avec[left_idx]==bvec[right_idx])
			{
				int32_t old_left_idx=left_idx;
				int32_t old_right_idx=right_idx;

				uint64_t sym=avec[left_idx];

				while (left_idx< alen && avec[left_idx]==sym)
					left_idx++;

				while (right_idx< blen && bvec[right_idx]==sym)
					right_idx++;

				result+=((float64_t) (left_idx-old_left_idx)) * ((float64_t) (right_idx-old_right_idx));
			}
			else if (avec[left_idx]<bvec[right_idx])
				left_idx++;
			else
				right_idx++;
		}
	}
	(std::static_pointer_cast<StringFeatures<uint64_t>>(lhs))->free_feature_vector(avec, idx_a, free_avec);
	(std::static_pointer_cast<StringFeatures<uint64_t>>(rhs))->free_feature_vector(bvec, idx_b, free_bvec);

	return result;
}

void CommUlongStringKernel::add_to_normal(int32_t vec_idx, float64_t weight)
{
	int32_t t=0;
	int32_t j=0;
	int32_t k=0;
	int32_t last_j=0;
	int32_t len=-1;
	bool free_vec;
	uint64_t* vec=(std::static_pointer_cast<StringFeatures<uint64_t>>(lhs))->get_feature_vector(vec_idx, len, free_vec);

	if (vec && len>0)
	{
		int32_t max_len = len+dictionary.vlen;
		SGVector<uint64_t> dic(max_len);
		SGVector<float64_t> dic_weights(max_len);

		if (use_sign)
		{
			for (j=1; j<len; j++)
			{
				if (vec[j]==vec[j-1])
					continue;

				merge_dictionaries(t, j, k, vec, dic, dic_weights, weight, vec_idx);
			}

			merge_dictionaries(t, j, k, vec, dic, dic_weights, weight, vec_idx);

			while (k<dictionary.vlen)
			{
				dic[t]=dictionary[k];
				dic_weights[t]=dictionary_weights[k];
				t++;
				k++;
			}
		}
		else
		{
			for (j=1; j<len; j++)
			{
				if (vec[j]==vec[j-1])
					continue;

				merge_dictionaries(t, j, k, vec, dic, dic_weights, weight*(j-last_j), vec_idx);
				last_j = j;
			}

			merge_dictionaries(t, j, k, vec, dic, dic_weights, weight*(j-last_j), vec_idx);

			while (k<dictionary.vlen)
			{
				dic[t]=dictionary[k];
				dic_weights[t]=dictionary_weights[k];
				t++;
				k++;
			}
		}

		dic.resize_vector(t);
		dic_weights.resize_vector(t);
		dictionary = dic;
		dictionary_weights = dic_weights;
	}
	(std::static_pointer_cast<StringFeatures<uint64_t>>(lhs))->free_feature_vector(vec, vec_idx, free_vec);

	set_is_initialized(true);
}

void CommUlongStringKernel::clear_normal()
{
	dictionary.resize_vector(0);
	dictionary_weights.resize_vector(0);
	set_is_initialized(false);
}

bool CommUlongStringKernel::init_optimization(
	int32_t count, int32_t *IDX, float64_t * weights)
{
	clear_normal();

	if (count<=0)
	{
		set_is_initialized(true);
		SG_DEBUG("empty set of SVs")
		return true;
	}

	SG_DEBUG("initializing CCommUlongStringKernel optimization")

	for (auto i : SG_PROGRESS(range(0, count)))
	{
		add_to_normal(IDX[i], weights[i]);
	}

	SG_DEBUG("Done.         ")

	set_is_initialized(true);
	return true;
}

bool CommUlongStringKernel::delete_optimization()
{
	SG_DEBUG("deleting CCommUlongStringKernel optimization")
	clear_normal();
	return true;
}

// binary search for each feature. trick: as features are sorted save last found idx in old_idx and
// only search in the remainder of the dictionary
float64_t CommUlongStringKernel::compute_optimized(int32_t i)
{
	float64_t result = 0;
	int32_t j, last_j=0;
	int32_t old_idx = 0;

	if (!get_is_initialized())
	{
      error("CCommUlongStringKernel optimization not initialized");
		return 0 ;
	}



	int32_t alen = -1;
	bool free_avec;
	uint64_t* avec=(std::static_pointer_cast<StringFeatures<uint64_t>>(rhs))->
		get_feature_vector(i, alen, free_avec);

	if (avec && alen>0)
	{
		if (use_sign)
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue;

				int32_t idx = Math::binary_search_max_lower_equal(&(dictionary[old_idx]), dictionary.vlen-old_idx, avec[j-1]);

				if (idx!=-1)
				{
					if (dictionary[idx+old_idx] == avec[j-1])
						result += dictionary_weights[idx+old_idx];

					old_idx+=idx;
				}
			}

			int32_t idx = Math::binary_search(&(dictionary[old_idx]), dictionary.vlen-old_idx, avec[alen-1]);
			if (idx!=-1)
				result += dictionary_weights[idx+old_idx];
		}
		else
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue;

				int32_t idx = Math::binary_search_max_lower_equal(&(dictionary[old_idx]), dictionary.vlen-old_idx, avec[j-1]);

				if (idx!=-1)
				{
					if (dictionary[idx+old_idx] == avec[j-1])
						result += dictionary_weights[idx+old_idx]*(j-last_j);

					old_idx+=idx;
				}

				last_j = j;
			}

			int32_t idx = Math::binary_search(&(dictionary[old_idx]), dictionary.vlen-old_idx, avec[alen-1]);
			if (idx!=-1)
				result += dictionary_weights[idx+old_idx]*(alen-last_j);
		}
	}

	(std::static_pointer_cast<StringFeatures<uint64_t>>(rhs))->free_feature_vector(avec, i, free_avec);

	return normalizer->normalize_rhs(result, i);
}
