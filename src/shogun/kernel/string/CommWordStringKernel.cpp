/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Giovanni De Toni, Heiko Strathmann, Viktor Gal,
 *          Weijie Lin, Bjoern Esser, Sergey Lisitsyn
 */

#include <shogun/base/Parameter.h>
#include <shogun/base/progress.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>

#include <shogun/features/StringFeatures.h>
#include <shogun/kernel/string/CommWordStringKernel.h>

#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>

using namespace shogun;

CommWordStringKernel::CommWordStringKernel()
: StringKernel<uint16_t>()
{
	init();
}

CommWordStringKernel::CommWordStringKernel(int32_t size, bool s)
: StringKernel<uint16_t>(size)
{
	init();
	use_sign=s;
}

CommWordStringKernel::CommWordStringKernel(
	std::shared_ptr<StringFeatures<uint16_t>> l, std::shared_ptr<StringFeatures<uint16_t>> r,
	bool s, int32_t size) : StringKernel<uint16_t>(size)
{
	init();
	use_sign=s;

	init(l,r);
}


bool CommWordStringKernel::init_dictionary(int32_t size)
{
	dictionary_weights=SGVector<float64_t>(size);
	SG_DEBUG("using dictionary of %d words\n", size)
	clear_normal();

	return dictionary_weights.vector!=NULL;
}

CommWordStringKernel::~CommWordStringKernel()
{
	cleanup();

	SG_FREE(dict_diagonal_optimization);
}

bool CommWordStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	StringKernel<uint16_t>::init(l,r);

	if (use_dict_diagonal_optimization)
	{
		SG_FREE(dict_diagonal_optimization);
		auto sf_l = std::dynamic_pointer_cast<StringFeatures<uint16_t>>(l);
		auto sf_r = std::dynamic_pointer_cast<StringFeatures<uint16_t>>(r);
		dict_diagonal_optimization=SG_MALLOC(int32_t, int32_t(sf_l->get_num_symbols()));
		ASSERT(sf_l->get_num_symbols() == sf_r->get_num_symbols())
	}

	return init_normalizer();
}

void CommWordStringKernel::cleanup()
{
	delete_optimization();
	Kernel::cleanup();
}

float64_t CommWordStringKernel::compute_diag(int32_t idx_a)
{
	int32_t alen;
	auto l = std::static_pointer_cast<StringFeatures<uint16_t>>(lhs);
	auto r = std::static_pointer_cast<StringFeatures<uint16_t>>(rhs);

	bool free_av;
	uint16_t* av=l->get_feature_vector(idx_a, alen, free_av);

	float64_t result=0.0 ;
	ASSERT(l==r)
	ASSERT(sizeof(uint16_t)<=sizeof(float64_t))
	ASSERT((1<<(sizeof(uint16_t)*8)) > alen)

	int32_t num_symbols=(int32_t) l->get_num_symbols();
	ASSERT(num_symbols<=dictionary_weights.vlen)

	int32_t* dic = dict_diagonal_optimization;
	memset(dic, 0, num_symbols*sizeof(int32_t));

	for (int32_t i=0; i<alen; i++)
		dic[av[i]]++;

	if (use_sign)
	{
		for (int32_t i=0; i<(int32_t) l->get_num_symbols(); i++)
		{
			if (dic[i]!=0)
				result++;
		}
	}
	else
	{
		for (int32_t i=0; i<num_symbols; i++)
		{
			if (dic[i]!=0)
				result+=dic[i]*dic[i];
		}
	}
	l->free_feature_vector(av, idx_a, free_av);

	return result;
}

float64_t CommWordStringKernel::compute_helper(
	int32_t idx_a, int32_t idx_b, bool do_sort)
{
	int32_t alen, blen;
	bool free_av, free_bv;

	auto l = std::static_pointer_cast<StringFeatures<uint16_t>>(lhs);
	auto r = std::static_pointer_cast<StringFeatures<uint16_t>>(rhs);

	uint16_t* av=l->get_feature_vector(idx_a, alen, free_av);
	uint16_t* bv=r->get_feature_vector(idx_b, blen, free_bv);

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

	int32_t left_idx=0;
	int32_t right_idx=0;

	if (use_sign)
	{
		while (left_idx < alen && right_idx < blen)
		{
			if (avec[left_idx]==bvec[right_idx])
			{
				uint16_t sym=avec[left_idx];

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

				uint16_t sym=avec[left_idx];

				while (left_idx< alen && avec[left_idx]==sym)
					left_idx++;

				while (right_idx< blen && bvec[right_idx]==sym)
					right_idx++;

				result+=((float64_t) (left_idx-old_left_idx))*
					((float64_t) (right_idx-old_right_idx));
			}
			else if (avec[left_idx]<bvec[right_idx])
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

	l->free_feature_vector(av, idx_a, free_av);
	r->free_feature_vector(bv, idx_b, free_bv);

	return result;
}

void CommWordStringKernel::add_to_normal(int32_t vec_idx, float64_t weight)
{
	int32_t len=-1;
	bool free_vec;
	uint16_t* vec=(std::static_pointer_cast<StringFeatures<uint16_t>>(lhs))->
		get_feature_vector(vec_idx, len, free_vec);

	if (len>0)
	{
		int32_t j, last_j=0;
		if (use_sign)
		{
			for (j=1; j<len; j++)
			{
				if (vec[j]==vec[j-1])
					continue;

				dictionary_weights[(int32_t) vec[j-1]]+=normalizer->
					normalize_lhs(weight, vec_idx);
			}

			dictionary_weights[(int32_t) vec[len-1]]+=normalizer->
				normalize_lhs(weight, vec_idx);
		}
		else
		{
			for (j=1; j<len; j++)
			{
				if (vec[j]==vec[j-1])
					continue;

				dictionary_weights[(int32_t) vec[j-1]]+=normalizer->
					normalize_lhs(weight*(j-last_j), vec_idx);
				last_j = j;
			}

			dictionary_weights[(int32_t) vec[len-1]]+=normalizer->
				normalize_lhs(weight*(len-last_j), vec_idx);
		}
		set_is_initialized(true);
	}

	(std::static_pointer_cast<StringFeatures<uint16_t>>(lhs))->free_feature_vector(vec, vec_idx, free_vec);
}

void CommWordStringKernel::clear_normal()
{
	dictionary_weights.zero();
	set_is_initialized(false);
}

bool CommWordStringKernel::init_optimization(
	int32_t count, int32_t* IDX, float64_t* weights)
{
	delete_optimization();

	if (count<=0)
	{
		set_is_initialized(true);
		SG_DEBUG("empty set of SVs\n")
		return true;
	}

	SG_DEBUG("initializing CCommWordStringKernel optimization\n")

	for (auto i : SG_PROGRESS(range(0, count)))
	{
		add_to_normal(IDX[i], weights[i]);
	}

	set_is_initialized(true);
	return true;
}

bool CommWordStringKernel::delete_optimization()
{
	SG_DEBUG("deleting CCommWordStringKernel optimization\n")

	clear_normal();
	return true;
}

float64_t CommWordStringKernel::compute_optimized(int32_t i)
{
	if (!get_is_initialized())
	{
      SG_ERROR("CCommWordStringKernel optimization not initialized\n")
		return 0 ;
	}

	float64_t result = 0;
	int32_t len = -1;
	bool free_vec;
	uint16_t* vec=(std::static_pointer_cast<StringFeatures<uint16_t>>(rhs))->
		get_feature_vector(i, len, free_vec);

	int32_t j, last_j=0;
	if (vec && len>0)
	{
		if (use_sign)
		{
			for (j=1; j<len; j++)
			{
				if (vec[j]==vec[j-1])
					continue;

				result += dictionary_weights[(int32_t) vec[j-1]];
			}

			result += dictionary_weights[(int32_t) vec[len-1]];
		}
		else
		{
			for (j=1; j<len; j++)
			{
				if (vec[j]==vec[j-1])
					continue;

				result += dictionary_weights[(int32_t) vec[j-1]]*(j-last_j);
				last_j = j;
			}

			result += dictionary_weights[(int32_t) vec[len-1]]*(len-last_j);
		}

		result=normalizer->normalize_rhs(result, i);
	}
	(std::static_pointer_cast<StringFeatures<uint16_t>>(rhs))->free_feature_vector(vec, i, free_vec);
	return result;
}

float64_t* CommWordStringKernel::compute_scoring(
	int32_t max_degree, int32_t& num_feat, int32_t& num_sym, float64_t* target,
	int32_t num_suppvec, int32_t* IDX, float64_t* alphas, bool do_init)
{
	ASSERT(lhs)
	auto str=(std::static_pointer_cast<StringFeatures<uint16_t>>(lhs));
	num_feat=1;//str->get_max_vector_length();
	auto alpha=str->get_alphabet();
	ASSERT(alpha)
	int32_t num_bits=alpha->get_num_bits();
	int32_t order=str->get_order();
	ASSERT(max_degree<=order)
	//int32_t num_words=(int32_t) str->get_num_symbols();
	int32_t num_words=(int32_t) str->get_original_num_symbols();
	int32_t offset=0;

	num_sym=0;

	for (int32_t i=0; i<order; i++)
		num_sym+=Math::pow((int32_t) num_words,i+1);

	SG_DEBUG("num_words:%d, order:%d, len:%d sz:%d (len*sz:%d)\n", num_words, order,
			num_feat, num_sym, num_feat*num_sym);

	if (!target)
		target=SG_MALLOC(float64_t, num_feat*num_sym);
	memset(target, 0, num_feat*num_sym*sizeof(float64_t));

	if (do_init)
		init_optimization(num_suppvec, IDX, alphas);

	uint32_t kmer_mask=0;
	uint32_t words=Math::pow((int32_t) num_words,(int32_t) order);

	for (int32_t o=0; o<max_degree; o++)
	{
		float64_t* contrib=&target[offset];
		offset+=Math::pow((int32_t) num_words,(int32_t) o+1);

		kmer_mask=(kmer_mask<<(num_bits)) | str->get_masked_symbols(0xffff, 1);

		for (int32_t p=-o; p<order; p++)
		{
			int32_t o_sym=0, m_sym=0, il=0,ir=0, jl=0;
			uint32_t imer_mask=kmer_mask;
			uint32_t jmer_mask=kmer_mask;

			if (p<0)
			{
				il=-p;
				m_sym=order-o-p-1;
				o_sym=-p;
			}
			else if (p<order-o)
			{
				ir=p;
				m_sym=order-o-1;
			}
			else
			{
				ir=p;
				m_sym=p;
				o_sym=p-order+o+1;
				jl=order-ir;
				imer_mask=(kmer_mask>>(num_bits*o_sym));
				jmer_mask=(kmer_mask>>(num_bits*jl));
			}

			float64_t marginalizer=
				1.0/Math::pow((int32_t) num_words,(int32_t) m_sym);

			for (uint32_t i=0; i<words; i++)
			{
				uint16_t x= ((i << (num_bits*il)) >> (num_bits*ir)) & imer_mask;

				if (p>=0 && p<order-o)
				{
//#define DEBUG_COMMSCORING
#ifdef DEBUG_COMMSCORING
					SG_PRINT("o=%d/%d p=%d/%d i=0x%x x=0x%x imask=%x jmask=%x kmask=%x il=%d ir=%d marg=%g o_sym:%d m_sym:%d weight(",
							o,order, p,order, i, x, imer_mask, jmer_mask, kmer_mask, il, ir, marginalizer, o_sym, m_sym);

					SG_PRINT("%c%c%c%c/%c%c%c%c)+=%g/%g\n",
							alpha->remap_to_char((x>>(3*num_bits))&0x03), alpha->remap_to_char((x>>(2*num_bits))&0x03),
							alpha->remap_to_char((x>>num_bits)&0x03), alpha->remap_to_char(x&0x03),
							alpha->remap_to_char((i>>(3*num_bits))&0x03), alpha->remap_to_char((i>>(2*num_bits))&0x03),
							alpha->remap_to_char((i>>(1*num_bits))&0x03), alpha->remap_to_char(i&0x03),
							dictionary_weights[i]*marginalizer, dictionary_weights[i]);
#endif
					contrib[x]+=dictionary_weights[i]*marginalizer;
				}
				else
				{
					for (uint32_t j=0; j< (uint32_t) Math::pow((int32_t) num_words, (int32_t) o_sym); j++)
					{
						uint32_t c=x | ((j & jmer_mask) << (num_bits*jl));
#ifdef DEBUG_COMMSCORING

						SG_PRINT("o=%d/%d p=%d/%d i=0x%x j=0x%x x=0x%x c=0x%x imask=%x jmask=%x kmask=%x il=%d ir=%d jl=%d marg=%g o_sym:%d m_sym:%d weight(",
								o,order, p,order, i, j, x, c, imer_mask, jmer_mask, kmer_mask, il, ir, jl, marginalizer, o_sym, m_sym);
						SG_PRINT("%c%c%c%c/%c%c%c%c)+=%g/%g\n",
								alpha->remap_to_char((c>>(3*num_bits))&0x03), alpha->remap_to_char((c>>(2*num_bits))&0x03),
								alpha->remap_to_char((c>>num_bits)&0x03), alpha->remap_to_char(c&0x03),
								alpha->remap_to_char((i>>(3*num_bits))&0x03), alpha->remap_to_char((i>>(2*num_bits))&0x03),
								alpha->remap_to_char((i>>(1*num_bits))&0x03), alpha->remap_to_char(i&0x03),
								dictionary_weights[i]*marginalizer, dictionary_weights[i]);
#endif
						contrib[c]+=dictionary_weights[i]*marginalizer;
					}
				}
			}
		}
	}

	for (int32_t i=1; i<num_feat; i++)
		sg_memcpy(&target[num_sym*i], target, num_sym*sizeof(float64_t));



	return target;
}


char* CommWordStringKernel::compute_consensus(
	int32_t &result_len, int32_t num_suppvec, int32_t* IDX, float64_t* alphas)
{
	ASSERT(lhs)
	ASSERT(IDX)
	ASSERT(alphas)

	auto str=(std::static_pointer_cast<StringFeatures<uint16_t>>(lhs));
	int32_t num_words=(int32_t) str->get_num_symbols();
	int32_t num_feat=str->get_max_vector_length();
	int64_t total_len=((int64_t) num_feat) * num_words;
	auto alpha=(std::static_pointer_cast<StringFeatures<uint16_t>>(lhs))->get_alphabet();
	ASSERT(alpha)
	int32_t num_bits=alpha->get_num_bits();
	int32_t order=str->get_order();
	int32_t max_idx=-1;
	float64_t max_score=0;
	result_len=num_feat+order-1;

	//init
	init_optimization(num_suppvec, IDX, alphas);

	char* result=SG_MALLOC(char, result_len);
	int32_t* bt=SG_MALLOC(int32_t, total_len);
	float64_t* score=SG_MALLOC(float64_t, total_len);

	for (int64_t i=0; i<total_len; i++)
	{
		bt[i]=-1;
		score[i]=0;
	}

	for (int32_t t=0; t<num_words; t++)
		score[t]=dictionary_weights[t];

	//dynamic program
	for (int32_t i=1; i<num_feat; i++)
	{
		for (int32_t t1=0; t1<num_words; t1++)
		{
			max_idx=-1;
			max_score=0;

			/* ignore weights the svm does not care about
			 * (has not seen in training). note that this assumes that zero
			 * weights are very unlikely to appear elsewise */

			//if (dictionary_weights[t1]==0.0)
				//continue;

			/* iterate over words t ending on t1 and find the highest scoring
			 * pair */
			uint16_t suffix=(uint16_t) t1 >> num_bits;

			for (int32_t sym=0; sym<str->get_original_num_symbols(); sym++)
			{
				uint16_t t=suffix | sym << (num_bits*(order-1));

				//if (dictionary_weights[t]==0.0)
				//	continue;

				float64_t sc=score[num_words*(i-1) + t]+dictionary_weights[t1];
				if (sc > max_score || max_idx==-1)
				{
					max_idx=t;
					max_score=sc;
				}
			}
			ASSERT(max_idx!=-1)

			score[num_words*i + t1]=max_score;
			bt[num_words*i + t1]=max_idx;
		}
	}

	//backtracking
	max_idx=0;
	max_score=score[num_words*(num_feat-1) + 0];
	for (int32_t t=1; t<num_words; t++)
	{
		float64_t sc=score[num_words*(num_feat-1) + t];
		if (sc>max_score)
		{
			max_idx=t;
			max_score=sc;
		}
	}

	SG_DEBUG("max_idx:%i, max_score:%f\n", max_idx, max_score)

	for (int32_t i=result_len-1; i>=num_feat; i--)
		result[i]=alpha->remap_to_char( (uint8_t) str->get_masked_symbols( (uint16_t) max_idx >> (num_bits*(result_len-1-i)), 1) );

	for (int32_t i=num_feat-1; i>=0; i--)
	{
		result[i]=alpha->remap_to_char( (uint8_t) str->get_masked_symbols( (uint16_t) max_idx >> (num_bits*(order-1)), 1) );
		max_idx=bt[num_words*i + max_idx];
	}

	SG_FREE(bt);
	SG_FREE(score);

	return result;
}

void CommWordStringKernel::init()
{
	use_sign=false;
	use_dict_diagonal_optimization=false;
	dict_diagonal_optimization=NULL;

	properties |= KP_LINADD;
	init_dictionary(1<<(sizeof(uint16_t)*8));
	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>(use_dict_diagonal_optimization));

	SG_ADD(&dictionary_weights,  "dictionary_weights",
			"Dictionary for applying kernel.");
	SG_ADD(&use_sign, "use_sign",
	    "If signum(counts) is used instead of counts.", ParameterProperties::HYPER);
	SG_ADD(&use_dict_diagonal_optimization,
	    "use_dict_diagonal_optimization", "If K(x,x) is computed potentially "
	    "more efficiently.");
}
