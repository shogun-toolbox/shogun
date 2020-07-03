/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Soeren Sonnenburg, Sergey Lisitsyn,
 *          Heiko Strathmann, Viktor Gal, Weijie Lin, Bjoern Esser
 */

#include <shogun/base/Parallel.h>
#include <shogun/base/progress.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Trie.h>
#include <shogun/lib/common.h>

#include <shogun/kernel/string/WeightedDegreePositionStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>

#include <shogun/classifier/svm/SVM.h>

#include <vector>

#ifdef HAVE_PTHREAD
#include <pthread.h>

#endif

using namespace shogun;

#define TRIES(X) ((use_poim_tries) ? (poim_tries->X) : (tries->X))

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <class Trie> struct S_THREAD_PARAM_WDS
{
	int32_t* vec;
	float64_t* result;
	float64_t* weights;
	WeightedDegreePositionStringKernel* kernel;
	CTrie<Trie>* tries;
	float64_t factor;
	int32_t j;
	int32_t start;
	int32_t end;
	int32_t length;
	int32_t max_shift;
	int32_t* shift;
	int32_t* vec_idx;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

WeightedDegreePositionStringKernel::WeightedDegreePositionStringKernel(
	void)
: StringKernel<char>()
{
	init();
}

WeightedDegreePositionStringKernel::WeightedDegreePositionStringKernel(
	int32_t size, int32_t d, int32_t mm, int32_t mkls)
: StringKernel<char>(size)
{
	init();

	mkl_stepsize=mkls;
	degree=d;
	max_mismatch=mm;

	set_wd_weights();

	tries=std::make_unique<CTrie<DNATrie>>(d);
	poim_tries=std::make_unique<CTrie<POIMTrie>>(d);
}

WeightedDegreePositionStringKernel::WeightedDegreePositionStringKernel(
	int32_t size, SGVector<float64_t> w, int32_t d, int32_t mm, SGVector<int32_t> s,
	int32_t mkls)
: StringKernel<char>(size)
{
	init();

	mkl_stepsize=mkls;
	degree=d;
	max_mismatch=mm;

	tries=std::make_unique<CTrie<DNATrie>>(d);
	poim_tries=std::make_unique<CTrie<POIMTrie>>(d);

	weights=w.clone();
	weights_degree=degree;
	weights_length=(1+max_mismatch);

	set_shifts(s);
}

WeightedDegreePositionStringKernel::WeightedDegreePositionStringKernel(
	const std::shared_ptr<Features>& l, const std::shared_ptr<Features>& r, int32_t d)
: StringKernel<char>()
{
	init();

	mkl_stepsize=1;
	degree=d;

	tries=std::make_unique<CTrie<DNATrie>>(d);
	poim_tries=std::make_unique<CTrie<POIMTrie>>(d);
	set_wd_weights();

	init(l->as<StringFeatures<char>>(), r->as<StringFeatures<char>>());
}


WeightedDegreePositionStringKernel::~WeightedDegreePositionStringKernel()
{
	cleanup();
	cleanup_POIM2();
}

void WeightedDegreePositionStringKernel::remove_lhs()
{
	SG_DEBUG("deleting CWeightedDegreePositionStringKernel optimization")
	delete_optimization();

	Kernel::remove_lhs();
}

void WeightedDegreePositionStringKernel::create_empty_tries()
{
	ASSERT(lhs)
	seq_length = std::static_pointer_cast<StringFeatures<char>>(lhs)->get_max_vector_length();

	if (opt_type==SLOWBUTMEMEFFICIENT)
	{
		tries->create(seq_length, true);
		poim_tries->create(seq_length, true);
	}
	else if (opt_type==FASTBUTMEMHUNGRY)
	{
		tries->create(seq_length, false);  // still buggy
		poim_tries->create(seq_length, false);  // still buggy
	}
	else
		error("unknown optimization type");
}

bool WeightedDegreePositionStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	set_wd_weights();

	int32_t lhs_changed = (lhs!=l) ;
	int32_t rhs_changed = (rhs!=r) ;

	StringKernel<char>::init(l,r);

	SG_DEBUG("lhs_changed: {}", lhs_changed)
	SG_DEBUG("rhs_changed: {}", rhs_changed)

	auto sf_l=std::static_pointer_cast<StringFeatures<char>>(l);
	auto sf_r=std::static_pointer_cast<StringFeatures<char>>(r);

	/* set shift */
	if (shift_len==0) {
		shift_len=sf_l->get_vector_length(0);
		auto shifts = SGVector<int32_t>(shift_len);
		shifts.set_const(1);
		set_shifts(shifts);
	}

	int32_t len=sf_l->get_max_vector_length();

	if (lhs_changed && !sf_l->have_same_length(len))
		error("All strings in WD kernel must have same length (lhs wrong)!");

	if (rhs_changed && !sf_r->have_same_length(len))
		error("All strings in WD kernel must have same length (rhs wrong)!");

	const auto& alphabet= sf_l->get_alphabet();
	const auto& ralphabet=sf_r->get_alphabet();

	if (!((alphabet->get_alphabet()==DNA) || (alphabet->get_alphabet()==RNA)))
		properties &= ((uint64_t) (-1)) ^ (KP_LINADD | KP_BATCHEVALUATION);

	ASSERT(ralphabet->get_alphabet()==alphabet->get_alphabet())

	//whenever init is called also init tries and block weights
	create_empty_tries();
	init_block_weights();

	return init_normalizer();
}

void WeightedDegreePositionStringKernel::cleanup()
{
	SG_DEBUG("deleting CWeightedDegreePositionStringKernel optimization")
	delete_optimization();

	seq_length = 0;
	tree_initialized = false;

	Kernel::cleanup();
}

bool WeightedDegreePositionStringKernel::init_optimization(
	int32_t p_count, int32_t * IDX, float64_t * alphas, int32_t tree_num,
	int32_t upto_tree)
{
	ASSERT(position_weights_lhs.size() > 0)
	ASSERT(position_weights_rhs.size() > 0)

	if (upto_tree<0)
		upto_tree=tree_num;

	if (max_mismatch!=0)
	{
		error("CWeightedDegreePositionStringKernel optimization not implemented for mismatch!=0");
		return false ;
	}

	if (tree_num<0)
		SG_DEBUG("deleting CWeightedDegreePositionStringKernel optimization")

	delete_optimization();

	if (tree_num<0)
		SG_DEBUG("initializing CWeightedDegreePositionStringKernel optimization")

	for (auto i : SG_PROGRESS(range(p_count)))
	{
		if (tree_num<0)
		{
			add_example_to_tree(IDX[i], alphas[i]);
		}
		else
		{
			for (int32_t t=tree_num; t<=upto_tree; t++)
				add_example_to_single_tree(IDX[i], alphas[i], t);
		}
	}

	set_is_initialized(true) ;
	return true ;
}

bool WeightedDegreePositionStringKernel::delete_optimization()
{
	if ((opt_type==FASTBUTMEMHUNGRY) && (tries->get_use_compact_terminal_nodes()))
	{
		tries->set_use_compact_terminal_nodes(false) ;
		SG_DEBUG("disabling compact trie nodes with FASTBUTMEMHUNGRY")
	}

	if (get_is_initialized())
	{
		if (opt_type==SLOWBUTMEMEFFICIENT)
			tries->delete_trees(true);
		else if (opt_type==FASTBUTMEMHUNGRY)
			tries->delete_trees(false);  // still buggy
		else {
			error("unknown optimization type");
		}
		set_is_initialized(false);

		return true;
	}

	return false;
}

float64_t WeightedDegreePositionStringKernel::compute_with_mismatch(
	char* avec, int32_t alen, char* bvec, int32_t blen)
{
	SGVector<float64_t> max_shift_vec(max_shift);
	max_shift_vec.zero();
	float64_t sum0=0;

    // no shift
    for (int32_t i=0; i<alen; i++)
    {
		if ((position_weights.size() > 0) && (position_weights[i]==0.0))
			continue ;

		int32_t mismatches=0;
		float64_t sumi = 0.0 ;
		for (int32_t j=0; (j<degree) && (i+j<alen); j++)
		{
			if (avec[i+j]!=bvec[i+j])
			{
				mismatches++ ;
				if (mismatches>max_mismatch)
					break ;
			} ;
			sumi += weights[j+degree*mismatches];
		}
		if (position_weights.size() > 0)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
    } ;

    for (int32_t i=0; i<alen; i++)
    {
		for (int32_t k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			if ((position_weights.size() > 0) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;

			float64_t sumi1 = 0.0 ;
			// shift in sequence a
			int32_t mismatches=0;
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
				{
					mismatches++ ;
					if (mismatches>max_mismatch)
						break ;
				} ;
				sumi1 += weights[j+degree*mismatches];
			}
			float64_t sumi2 = 0.0 ;
			// shift in sequence b
			mismatches=0;
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j]!=bvec[i+j+k])
				{
					mismatches++ ;
					if (mismatches>max_mismatch)
						break ;
				} ;
				sumi2 += weights[j+degree*mismatches];
			}
			if (position_weights.size() > 0)
				max_shift_vec[k-1] += position_weights[i]*sumi1 + position_weights[i+k]*sumi2 ;
			else
				max_shift_vec[k-1] += sumi1 + sumi2 ;
		} ;
    }

    float64_t result = sum0 ;
    for (int32_t i=0; i<max_shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;

    return result ;
}

float64_t WeightedDegreePositionStringKernel::compute_without_mismatch(
	char* avec, int32_t alen, char* bvec, int32_t blen)
{
	SGVector<float64_t> max_shift_vec(max_shift);
	max_shift_vec.zero();
	float64_t sum0=0 ;

	// no shift
	for (int32_t i=0; i<alen; i++)
	{
		if ((position_weights.size() > 0) && (position_weights[i]==0.0))
			continue ;

		float64_t sumi = 0.0 ;
		for (int32_t j=0; (j<degree) && (i+j<alen); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break ;
			sumi += weights[j];
		}
		if (position_weights.size() > 0)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
	} ;

	for (int32_t i=0; i<alen; i++)
	{
		for (int32_t k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			if ((position_weights.size() > 0) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;

			float64_t sumi1 = 0.0 ;
			// shift in sequence a
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi1 += weights[j];
			}
			float64_t sumi2 = 0.0 ;
			// shift in sequence b
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j]!=bvec[i+j+k])
					break ;
				sumi2 += weights[j];
			}
			if (position_weights.size() > 0)
				max_shift_vec[k-1] += position_weights[i]*sumi1 + position_weights[i+k]*sumi2 ;
			else
				max_shift_vec[k-1] += sumi1 + sumi2 ;
		} ;
	}

	float64_t result = sum0 ;
	for (int32_t i=0; i<max_shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;

	return result ;
}

float64_t WeightedDegreePositionStringKernel::compute_without_mismatch_matrix(
	char* avec, int32_t alen, char* bvec, int32_t blen)
{
	SGVector<float64_t> max_shift_vec(max_shift);
	max_shift_vec.zero();
	float64_t sum0=0 ;

	// no shift
	for (int32_t i=0; i<alen; i++)
	{
		if ((position_weights.size() > 0) && (position_weights[i]==0.0))
			continue ;
		float64_t sumi = 0.0 ;
		for (int32_t j=0; (j<degree) && (i+j<alen); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break ;
			sumi += weights[i*degree+j];
		}
		if (position_weights.size() > 0)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
	} ;

	for (int32_t i=0; i<alen; i++)
	{
		for (int32_t k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			if ((position_weights.size() > 0) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;

			float64_t sumi1 = 0.0 ;
			// shift in sequence a
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi1 += weights[i*degree+j];
			}
			float64_t sumi2 = 0.0 ;
			// shift in sequence b
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j]!=bvec[i+j+k])
					break ;
				sumi2 += weights[i*degree+j];
			}
			if (position_weights.size() > 0)
				max_shift_vec[k-1] += position_weights[i]*sumi1 + position_weights[i+k]*sumi2 ;
			else
				max_shift_vec[k-1] += sumi1 + sumi2 ;
		} ;
	}

	float64_t result = sum0 ;
	for (int32_t i=0; i<max_shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;

	return result ;
}

float64_t WeightedDegreePositionStringKernel::compute_without_mismatch_position_weights(
	char* avec, float64_t* pos_weights_lhs, int32_t alen, char* bvec,
	float64_t* pos_weights_rhs, int32_t blen)
{
	SGVector<float64_t> max_shift_vec(max_shift);
	max_shift_vec.zero();
	float64_t sum0=0;

	// no shift
	for (int32_t i=0; i<alen; i++)
	{
		if ((position_weights.size() > 0) && (position_weights[i]==0.0))
			continue ;

		float64_t sumi = 0.0 ;
	    float64_t posweight_lhs = 0.0 ;
	    float64_t posweight_rhs = 0.0 ;
		for (int32_t j=0; (j<degree) && (i+j<alen); j++)
		{
			posweight_lhs += pos_weights_lhs[i+j] ;
			posweight_rhs += pos_weights_rhs[i+j] ;

			if (avec[i+j]!=bvec[i+j])
				break ;
			sumi += weights[j]*(posweight_lhs/(j+1))*(posweight_rhs/(j+1)) ;
		}
		if (position_weights.size() > 0)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
	} ;

	for (int32_t i=0; i<alen; i++)
	{
		for (int32_t k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			if ((position_weights.size() > 0) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;

			// shift in sequence a
			float64_t sumi1 = 0.0 ;
			float64_t posweight_lhs = 0.0 ;
			float64_t posweight_rhs = 0.0 ;
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
			{
				posweight_lhs += pos_weights_lhs[i+j+k] ;
				posweight_rhs += pos_weights_rhs[i+j] ;
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi1 += weights[j]*(posweight_lhs/(j+1))*(posweight_rhs/(j+1)) ;
			}
			// shift in sequence b
			float64_t sumi2 = 0.0 ;
			posweight_lhs = 0.0 ;
			posweight_rhs = 0.0 ;
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
			{
				posweight_lhs += pos_weights_lhs[i+j] ;
				posweight_rhs += pos_weights_rhs[i+j+k] ;
				if (avec[i+j]!=bvec[i+j+k])
					break ;
				sumi2 += weights[j]*(posweight_lhs/(j+1))*(posweight_rhs/(j+1)) ;
			}
			if (position_weights.size() > 0)
				max_shift_vec[k-1] += position_weights[i]*sumi1 + position_weights[i+k]*sumi2 ;
			else
				max_shift_vec[k-1] += sumi1 + sumi2 ;
		} ;
	}

	float64_t result = sum0 ;
	for (int32_t i=0; i<max_shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;

	return result ;
}


float64_t WeightedDegreePositionStringKernel::compute(
	int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	char* avec=std::static_pointer_cast<StringFeatures<char>>(lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec=std::static_pointer_cast<StringFeatures<char>>(rhs)->get_feature_vector(idx_b, blen, free_bvec);
	// can only deal with strings of same length
	ASSERT(alen==blen)
	ASSERT(shift_len==alen)

	float64_t result = 0 ;
	if (position_weights_lhs.size() > 0 || position_weights_rhs.size() > 0)
	{
		ASSERT(max_mismatch==0)
		auto position_weights_rhs_ = position_weights_rhs ;
		if (lhs==rhs)
			position_weights_rhs_ = position_weights_lhs ;

		result = compute_without_mismatch_position_weights(avec, &position_weights_lhs[idx_a*alen], alen, bvec, &position_weights_rhs_[idx_b*blen], blen) ;
	}
	else if (max_mismatch > 0)
		result = compute_with_mismatch(avec, alen, bvec, blen) ;
	else if (length==0)
		result = compute_without_mismatch(avec, alen, bvec, blen) ;
	else
		result = compute_without_mismatch_matrix(avec, alen, bvec, blen) ;

	std::static_pointer_cast<StringFeatures<char>>(lhs)->free_feature_vector(avec, idx_a, free_avec);
	std::static_pointer_cast<StringFeatures<char>>(rhs)->free_feature_vector(bvec, idx_b, free_bvec);

	return result ;
}


void WeightedDegreePositionStringKernel::add_example_to_tree(
	int32_t idx, float64_t alpha)
{
	ASSERT(position_weights_lhs.size() > 0)
	ASSERT(position_weights_rhs.size() > 0)
	auto alphabet = std::static_pointer_cast<StringFeatures<char>>(lhs)->get_alphabet();
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	int32_t len=0;
	bool free_vec;
	char* char_vec=std::static_pointer_cast<StringFeatures<char>>(lhs)->get_feature_vector(idx, len, free_vec);
	ASSERT(max_mismatch==0)
	int32_t *vec = SG_MALLOC(int32_t, len);

	for (int32_t i=0; i<len; i++)
		vec[i]=alphabet->remap_to_bin(char_vec[i]);
	std::static_pointer_cast<StringFeatures<char>>(lhs)->free_feature_vector(char_vec, idx, free_vec);

	if (opt_type==FASTBUTMEMHUNGRY)
	{
		//TRIES(set_use_compact_terminal_nodes(false)) ;
		ASSERT(!TRIES(get_use_compact_terminal_nodes()))
	}

	for (int32_t i=0; i<len; i++)
	{
		int32_t max_s=-1;

		if (opt_type==SLOWBUTMEMEFFICIENT)
			max_s=0;
		else if (opt_type==FASTBUTMEMHUNGRY)
			max_s=shift[i];
		else {
			error("unknown optimization type");
		}

		for (int32_t s=max_s; s>=0; s--)
		{
			float64_t alpha_pw = normalizer->normalize_lhs((s==0) ? (alpha) : (alpha/(2.0*s)), idx);
			TRIES(add_to_trie(i, s, vec, alpha_pw, weights.vector, (length!=0))) ;
			if ((s==0) || (i+s>=len))
				continue;

			TRIES(add_to_trie(i+s, -s, vec, alpha_pw, weights.vector, (length!=0))) ;
		}
	}

	SG_FREE(vec);
	tree_initialized=true ;
}

void WeightedDegreePositionStringKernel::add_example_to_single_tree(
	int32_t idx, float64_t alpha, int32_t tree_num)
{
	ASSERT(position_weights_lhs.size() > 0)
	ASSERT(position_weights_rhs.size() > 0)
	auto alphabet = std::static_pointer_cast<StringFeatures<char>>(lhs)->get_alphabet();
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	int32_t len=0;
	bool free_vec;
	char* char_vec=std::static_pointer_cast<StringFeatures<char>>(lhs)->get_feature_vector(idx, len, free_vec);
	ASSERT(max_mismatch==0)
	int32_t *vec=SG_MALLOC(int32_t, len);
	int32_t max_s=-1;

	if (opt_type==SLOWBUTMEMEFFICIENT)
		max_s=0;
	else if (opt_type==FASTBUTMEMHUNGRY)
	{
		ASSERT(!tries->get_use_compact_terminal_nodes())
		max_s=shift[tree_num];
	}
	else {
		error("unknown optimization type");
	}
	for (int32_t i=Math::max(0,tree_num-max_shift);
			i<Math::min(len,tree_num+degree+max_shift); i++)
	{
		vec[i]=alphabet->remap_to_bin(char_vec[i]);
	}
	std::static_pointer_cast<StringFeatures<char>>(lhs)->free_feature_vector(char_vec, idx, free_vec);

	for (int32_t s=max_s; s>=0; s--)
	{
		float64_t alpha_pw = normalizer->normalize_lhs((s==0) ? (alpha) : (alpha/(2.0*s)), idx);
		tries->add_to_trie(tree_num, s, vec, alpha_pw, weights.vector, (length!=0)) ;
	}

	if (opt_type==FASTBUTMEMHUNGRY)
	{
		for (int32_t i=Math::max(0,tree_num-max_shift); i<Math::min(len,tree_num+max_shift+1); i++)
		{
			int32_t s=tree_num-i;
			if ((i+s<len) && (s>=1) && (s<=shift[i]))
			{
				float64_t alpha_pw = normalizer->normalize_lhs((s==0) ? (alpha) : (alpha/(2.0*s)), idx);
				tries->add_to_trie(tree_num, -s, vec, alpha_pw, weights.vector, (length!=0)) ;
			}
		}
	}
	SG_FREE(vec);
	tree_initialized=true ;
}

float64_t WeightedDegreePositionStringKernel::compute_by_tree(int32_t idx)
{
	ASSERT(position_weights_lhs.size() > 0)
	ASSERT(position_weights_rhs.size() > 0)
	auto alphabet = std::static_pointer_cast<StringFeatures<char>>(rhs)->get_alphabet();
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	float64_t sum=0;
	int32_t len=0;
	bool free_vec;
	char* char_vec=std::static_pointer_cast<StringFeatures<char>>(rhs)->get_feature_vector(idx, len, free_vec);
	ASSERT(max_mismatch==0)
	int32_t *vec=SG_MALLOC(int32_t, len);

	for (int32_t i=0; i<len; i++)
		vec[i]=alphabet->remap_to_bin(char_vec[i]);

	std::static_pointer_cast<StringFeatures<char>>(rhs)->free_feature_vector(char_vec, idx, free_vec);

	for (int32_t i=0; i<len; i++)
		sum += tries->compute_by_tree_helper(vec, len, i, i, i, weights.vector, (length!=0)) ;

	if (opt_type==SLOWBUTMEMEFFICIENT)
	{
		for (int32_t i=0; i<len; i++)
		{
			for (int32_t s=1; (s<=shift[i]) && (i+s<len); s++)
			{
				sum+=tries->compute_by_tree_helper(vec, len, i, i+s, i, weights.vector, (length!=0))/(2*s) ;
				sum+=tries->compute_by_tree_helper(vec, len, i+s, i, i+s, weights.vector, (length!=0))/(2*s) ;
			}
		}
	}

	SG_FREE(vec);

	return normalizer->normalize_rhs(sum, idx);
}

void WeightedDegreePositionStringKernel::compute_by_tree(
	int32_t idx, float64_t* LevelContrib)
{
	ASSERT(position_weights_lhs.size() > 0)
	ASSERT(position_weights_rhs.size() > 0)
	auto alphabet = std::static_pointer_cast<StringFeatures<char>>(rhs)->get_alphabet();
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	int32_t len=0;
	bool free_vec;
	char* char_vec=std::static_pointer_cast<StringFeatures<char>>(rhs)->get_feature_vector(idx, len, free_vec);
	ASSERT(max_mismatch==0)
	int32_t *vec=SG_MALLOC(int32_t, len);

	for (int32_t i=0; i<len; i++)
		vec[i]=alphabet->remap_to_bin(char_vec[i]);

	std::static_pointer_cast<StringFeatures<char>>(rhs)->free_feature_vector(char_vec, idx, free_vec);

	for (int32_t i=0; i<len; i++)
	{
		tries->compute_by_tree_helper(vec, len, i, i, i, LevelContrib,
				normalizer->normalize_rhs(1.0, idx), mkl_stepsize, weights.vector,
				(length!=0));
	}

	if (opt_type==SLOWBUTMEMEFFICIENT)
	{
		for (int32_t i=0; i<len; i++)
			for (int32_t k=1; (k<=shift[i]) && (i+k<len); k++)
			{
				tries->compute_by_tree_helper(vec, len, i, i+k, i, LevelContrib,
						normalizer->normalize_rhs(1.0/(2*k), idx), mkl_stepsize,
						weights.vector, (length!=0)) ;
				tries->compute_by_tree_helper(vec, len, i+k, i, i+k,
						LevelContrib, normalizer->normalize_rhs(1.0/(2*k), idx),
						mkl_stepsize, weights.vector, (length!=0)) ;
			}
	}

	SG_FREE(vec);
}

float64_t* WeightedDegreePositionStringKernel::compute_abs_weights(
	int32_t &len)
{
	return tries->compute_abs_weights(len);
}

void WeightedDegreePositionStringKernel::set_shifts(SGVector<int32_t> shifts)
{
	shift = shifts;
	max_shift = *std::max_element(shift.begin(), shift.end());

	ASSERT(max_shift>=0 && max_shift<=shift_len)
}

bool WeightedDegreePositionStringKernel::set_wd_weights()
{
	ASSERT(degree>0)

	weights=SGVector<float64_t>(degree);
	weights_degree=degree;
	weights_length=1;

	int32_t i;
	float64_t sum=0;
	for (i=0; i<degree; i++)
	{
		weights[i]=degree-i;
		sum+=weights[i];
	}
	for (i=0; i<degree; i++)
		weights[i]/=sum;

	for (i=0; i<degree; i++)
	{
		for (int32_t j=1; j<=max_mismatch; j++)
		{
			if (j<i+1)
			{
				int32_t nk=Math::nchoosek(i+1, j);
				weights[i+j*degree]=weights[i]/(nk*Math::pow(3.0,j));
			}
			else
				weights[i+j*degree]= 0;
		}
	}
	return true;
}

bool WeightedDegreePositionStringKernel::set_weights(SGMatrix<float64_t> new_weights)
{
	int32_t d=new_weights.num_rows;
	int32_t len=new_weights.num_cols;

	if (d!=degree || len<0)
		error("WD: Dimension mismatch (should be (seq_length | 1) x degree) got ({} x {})", len, degree);

	degree=d;
	length=len;

	if (len <= 0)
		len=1;

	weights_degree=degree;
	weights_length=len+max_mismatch;

	SG_DEBUG("Creating weights of size {}x{}", weights_degree, weights_length)
	weights=new_weights.clone();

	return true;
}

void WeightedDegreePositionStringKernel::set_position_weights(const SGVector<float64_t>& pws)
{
	if (seq_length==0)
		seq_length=pws.size();

	if (seq_length!=pws.size())
		error("seq_length = {}, position_weights_length={}", seq_length, pws.size());

	position_weights=pws.clone();
	tries->set_position_weights(position_weights.vector);
}

bool WeightedDegreePositionStringKernel::set_position_weights_lhs(const SGMatrix<float64_t>& pws)
{
	if (pws.num_rows==0)
	{
		position_weights_lhs = SGVector<float64_t>();
		return true;
	}

	if (seq_length!=pws.num_rows)
	{
		error("seq_length = {}, position_weights_length={}", seq_length, pws.num_rows);
		return false;
	}

	position_weights_lhs=pws.clone();

	return true;
}

bool WeightedDegreePositionStringKernel::set_position_weights_rhs(
	const SGMatrix<float64_t>& pws)
{
	if (pws.num_rows==0)
	{
		position_weights_rhs=SGVector<float64_t>();
		return true;
	}

	if (seq_length!=pws.num_rows)
	{
		error("seq_length = {}, position_weights_length={}", seq_length, pws.num_rows);
		return false;
	}

	position_weights_rhs=SGMatrix<float64_t>::empty_like(pws);
	std::copy(pws.begin(), pws.end(), position_weights_rhs.begin());

	return true;
}

bool WeightedDegreePositionStringKernel::init_block_weights_from_wd()
{
	block_weights=SGVector<float64_t>(std::max(seq_length,degree));

	if (block_weights.vlen)
	{
		int32_t k;
		float64_t d=degree; // use float to evade rounding errors below

		for (k=0; k<degree; k++)
			block_weights[k]=
				(-Math::pow(k, 3)+(3*d-3)*Math::pow(k, 2)+(9*d-2)*k+6*d)/(3*d*(d+1));
		for (k=degree; k<seq_length; k++)
			block_weights[k]=(-d+3*k+4)/3;
	}

	return (block_weights.vlen > 0);
}

bool WeightedDegreePositionStringKernel::init_block_weights_from_wd_external()
{
	block_weights=SGVector<float64_t>(std::max(seq_length,degree));

	if (block_weights.vlen)
	{
		int32_t i=0;
		block_weights[0]=weights[0];
		for (i=1; i<Math::max(seq_length,degree); i++)
			block_weights[i]=0;

		for (i=1; i<Math::max(seq_length,degree); i++)
		{
			block_weights[i]=block_weights[i-1];

			float64_t contrib=0;
			for (int32_t j=0; j<Math::min(degree,i+1); j++)
				contrib+=weights[j];

			block_weights[i]+=contrib;
		}
	}

	return (block_weights.vlen > 0);
}

bool WeightedDegreePositionStringKernel::init_block_weights_const()
{
	block_weights=SGVector<float64_t>(seq_length);

	if (block_weights.vlen)
	{
		for (int32_t i=1; i<seq_length+1 ; i++)
			block_weights[i-1]=1.0/seq_length;
	}

	return (block_weights.vlen > 0);
}

bool WeightedDegreePositionStringKernel::init_block_weights_linear()
{
	block_weights=SGVector<float64_t>(seq_length);

	if (block_weights.vlen)
	{
		for (int32_t i=1; i<seq_length+1 ; i++)
			block_weights[i-1]=degree*i;
	}

	return (block_weights.vlen > 0);
}

bool WeightedDegreePositionStringKernel::init_block_weights_sqpoly()
{
	block_weights=SGVector<float64_t>(seq_length);

	if (block_weights.vlen)
	{
		for (int32_t i=1; i<degree+1 ; i++)
			block_weights[i-1]=((float64_t) i)*i;

		for (int32_t i=degree+1; i<seq_length+1 ; i++)
			block_weights[i-1]=i;
	}

	return (block_weights.vlen > 0);
}

bool WeightedDegreePositionStringKernel::init_block_weights_cubicpoly()
{
	block_weights=SGVector<float64_t>(seq_length);

	if (block_weights.vlen)
	{
		for (int32_t i=1; i<degree+1 ; i++)
			block_weights[i-1]=((float64_t) i)*i*i;

		for (int32_t i=degree+1; i<seq_length+1 ; i++)
			block_weights[i-1]=i;
	}

	return (block_weights.vlen > 0);
}

bool WeightedDegreePositionStringKernel::init_block_weights_exp()
{
	block_weights=SGVector<float64_t>(seq_length);

	if (block_weights.vlen)
	{
		for (int32_t i=1; i<degree+1 ; i++)
			block_weights[i-1]=exp(((float64_t) i/10.0));

		for (int32_t i=degree+1; i<seq_length+1 ; i++)
			block_weights[i-1]=i;
	}

	return (block_weights.vlen > 0);
}

bool WeightedDegreePositionStringKernel::init_block_weights_log()
{
	block_weights=SGVector<float64_t>(seq_length);

	if (block_weights.vlen)
	{
		for (int32_t i=1; i<degree+1 ; i++)
			block_weights[i - 1] = Math::pow(std::log((float64_t)i), 2);

		for (int32_t i=degree+1; i<seq_length+1 ; i++)
			block_weights[i - 1] =
			    i - degree + 1 + Math::pow(std::log(degree + 1.0), 2);
	}

	return (block_weights.vlen > 0);
}

bool WeightedDegreePositionStringKernel::init_block_weights()
{
	switch (type)
	{
		case E_WD:
			return init_block_weights_from_wd();
		case E_EXTERNAL:
			return init_block_weights_from_wd_external();
		case E_BLOCK_CONST:
			return init_block_weights_const();
		case E_BLOCK_LINEAR:
			return init_block_weights_linear();
		case E_BLOCK_SQPOLY:
			return init_block_weights_sqpoly();
		case E_BLOCK_CUBICPOLY:
			return init_block_weights_cubicpoly();
		case E_BLOCK_EXP:
			return init_block_weights_exp();
		case E_BLOCK_LOG:
			return init_block_weights_log();
	};
	return false;
}

void* WeightedDegreePositionStringKernel::compute_batch_helper(void* p)
{
	auto* params = (S_THREAD_PARAM_WDS<DNATrie>*) p;
	int32_t j=params->j;
	WeightedDegreePositionStringKernel* wd=params->kernel;
	CTrie<DNATrie>* tries=params->tries;
	float64_t* weights=params->weights;
	int32_t length=params->length;
	int32_t max_shift=params->max_shift;
	int32_t* vec=params->vec;
	float64_t* result=params->result;
	float64_t factor=params->factor;
	int32_t* shift=params->shift;
	int32_t* vec_idx=params->vec_idx;

	for (int32_t i=params->start; i<params->end; i++)
	{
		int32_t len=0;
		auto rhs_feat=(StringFeatures<char>*)(wd->get_rhs().get());
		auto alpha=rhs_feat->get_alphabet();

		bool free_vec;
		char* char_vec=rhs_feat->get_feature_vector(vec_idx[i], len, free_vec);
		for (int32_t k=Math::max(0,j-max_shift); k<Math::min(len,j+wd->get_degree()+max_shift); k++)
			vec[k]=alpha->remap_to_bin(char_vec[k]);
		rhs_feat->free_feature_vector(char_vec, vec_idx[i], free_vec);

		result[i] += factor*wd->normalizer->normalize_rhs(tries->compute_by_tree_helper(vec, len, j, j, j, weights, (length!=0)), vec_idx[i]);

		if (wd->get_optimization_type()==SLOWBUTMEMEFFICIENT)
		{
			for (int32_t q=Math::max(0,j-max_shift); q<Math::min(len,j+max_shift+1); q++)
			{
				int32_t s=j-q ;
				if ((s>=1) && (s<=shift[q]) && (q+s<len))
				{
					result[i] +=
						wd->normalizer->normalize_rhs(tries->compute_by_tree_helper(vec,
								len, q, q+s, q, weights, (length!=0)),
								vec_idx[i])/(2.0*s);
				}
			}

			for (int32_t s=1; (s<=shift[j]) && (j+s<len); s++)
			{
				result[i] +=
					wd->normalizer->normalize_rhs(tries->compute_by_tree_helper(vec,
								len, j+s, j, j+s, weights, (length!=0)),
								vec_idx[i])/(2.0*s);
			}
		}
	}

	return NULL;
}

void WeightedDegreePositionStringKernel::compute_batch(
	int32_t num_vec, int32_t* vec_idx, float64_t* result, int32_t num_suppvec,
	int32_t* IDX, float64_t* alphas, float64_t factor)
{
	auto alphabet = std::static_pointer_cast<StringFeatures<char>>(rhs)->get_alphabet();
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)
	ASSERT(rhs)
	ASSERT(num_vec<=rhs->get_num_vectors())
	ASSERT(num_vec>0)
	ASSERT(vec_idx)
	ASSERT(result)
	create_empty_tries();

	int32_t num_feat=std::static_pointer_cast<StringFeatures<char>>(rhs)->get_max_vector_length();
	ASSERT(num_feat>0)
	// TODO: port to use OpenMP backend instead of pthread
#ifdef HAVE_PTHREAD
	int32_t num_threads=env()->get_num_threads();
#else
	int32_t num_threads=1;
#endif
	ASSERT(num_threads>0)
	int32_t* vec=SG_MALLOC(int32_t, num_threads*num_feat);

	if (num_threads < 2)
	{
		// TODO: replace with the new signal
		// for (int32_t j=0; j<num_feat && !Signal::cancel_computations(); j++)
		for (auto j : SG_PROGRESS(range(num_feat)))
		{
			init_optimization(num_suppvec, IDX, alphas, j);
			S_THREAD_PARAM_WDS<DNATrie> params;
			params.vec = vec;
			params.result = result;
			params.weights = weights.vector;
			params.kernel = this;
			params.tries = tries.get();
			params.factor = factor;
			params.j = j;
			params.start = 0;
			params.end = num_vec;
			params.length = length;
			params.max_shift = max_shift;
			params.shift = shift;
			params.vec_idx = vec_idx;
			compute_batch_helper((void*)&params);
		    }
	}
#ifdef HAVE_PTHREAD
	else
	{
		// TODO: replace with the new signal
		// for (int32_t j=0; j<num_feat && !Signal::cancel_computations(); j++)
		for (auto j : SG_PROGRESS(range(num_feat)))
		{
			init_optimization(num_suppvec, IDX, alphas, j);
			pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
			std::vector<S_THREAD_PARAM_WDS<DNATrie>> params(num_threads);
			int32_t step= num_vec/num_threads;
			int32_t t;

			for (t=0; t<num_threads-1; t++)
			{
				params[t].vec=&vec[num_feat*t];
				params[t].result=result;
				params[t].weights=weights.vector;
				params[t].kernel=this;
				params[t].tries=tries.get();
				params[t].factor=factor;
				params[t].j=j;
				params[t].start = t*step;
				params[t].end = (t+1)*step;
				params[t].length=length;
				params[t].max_shift=max_shift;
				params[t].shift=shift;
				params[t].vec_idx=vec_idx;
				pthread_create(&threads[t], NULL, WeightedDegreePositionStringKernel::compute_batch_helper, (void*)&params[t]);
			}

			params[t].vec=&vec[num_feat*t];
			params[t].result=result;
			params[t].weights=weights.vector;
			params[t].kernel=this;
			params[t].tries=tries.get();
			params[t].factor=factor;
			params[t].j=j;
			params[t].start=t*step;
			params[t].end=num_vec;
			params[t].length=length;
			params[t].max_shift=max_shift;
			params[t].shift=shift;
			params[t].vec_idx=vec_idx;
			compute_batch_helper((void*) &params[t]);

			for (t=0; t<num_threads-1; t++)
				pthread_join(threads[t], NULL);

			SG_FREE(threads);
		}
	}
#endif

	SG_FREE(vec);

	//really also free memory as this can be huge on testing especially when
	//using the combined kernel
	create_empty_tries();
}

float64_t* WeightedDegreePositionStringKernel::compute_scoring(
	int32_t max_degree, int32_t& num_feat, int32_t& num_sym, float64_t* result,
	int32_t num_suppvec, int32_t* IDX, float64_t* alphas)
{
	num_feat=std::static_pointer_cast<StringFeatures<char>>(rhs)->get_max_vector_length();
	ASSERT(num_feat>0)
	auto alphabet = std::static_pointer_cast<StringFeatures<char>>(rhs)->get_alphabet();
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	num_sym=4; //for now works only w/ DNA

	ASSERT(max_degree>0)

	// === variables
	int32_t* nofsKmers=SG_MALLOC(int32_t, max_degree);
	float64_t** C=SG_MALLOC(float64_t*, max_degree);
	float64_t** L=SG_MALLOC(float64_t*, max_degree);
	float64_t** R=SG_MALLOC(float64_t*, max_degree);

	int32_t i;
	int32_t k;

	// --- return table
	int32_t bigtabSize=0;
	for (k=0; k<max_degree; ++k )
	{
		nofsKmers[k]=(int32_t) Math::pow(num_sym, k+1);
		const int32_t tabSize=nofsKmers[k]*num_feat;
		bigtabSize+=tabSize;
	}
	result=SG_MALLOC(float64_t, bigtabSize);

	// --- auxilliary tables
	int32_t tabOffs=0;
	for( k = 0; k < max_degree; ++k )
	{
		const int32_t tabSize = nofsKmers[k] * num_feat;
		C[k] = &result[tabOffs];
		L[k] = SG_MALLOC(float64_t,  tabSize );
		R[k] = SG_MALLOC(float64_t,  tabSize );
		tabOffs+=tabSize;
		for(i = 0; i < tabSize; i++ )
		{
			C[k][i] = 0.0;
			L[k][i] = 0.0;
			R[k][i] = 0.0;
		}
	}

	// --- tree parsing info
	float64_t* margFactors=SG_MALLOC(float64_t, degree);

	int32_t* x = SG_MALLOC(int32_t,  degree+1 );
	int32_t* substrs = SG_MALLOC(int32_t,  degree+1 );
	// - fill arrays
	margFactors[0] = 1.0;
	substrs[0] = 0;
	for( k=1; k < degree; ++k ) {
		margFactors[k] = 0.25 * margFactors[k-1];
		substrs[k] = -1;
	}
	substrs[degree] = -1;
	// - fill struct
	struct TreeParseInfo info;
	info.num_sym = num_sym;
	info.num_feat = num_feat;
	info.p = -1;
	info.k = -1;
	info.nofsKmers = nofsKmers;
	info.margFactors = margFactors;
	info.x = x;
	info.substrs = substrs;
	info.y0 = 0;
	info.C_k = NULL;
	info.L_k = NULL;
	info.R_k = NULL;

	// === main loop
	for (auto k : SG_PROGRESS(range(max_degree)))
	{
		const int32_t nofKmers = nofsKmers[ k ];
		info.C_k = C[k];
		info.L_k = L[k];
		info.R_k = R[k];

		// --- run over all trees
		for(int32_t p = 0; p < num_feat; ++p )
		{
			init_optimization( num_suppvec, IDX, alphas, p );
			int32_t tree = p ;
			for(int32_t j = 0; j < degree+1; j++ ) {
				x[j] = -1;
			}
			tries->traverse( tree, p, info, 0, x, k );
		}

		// --- add partial overlap scores
		if( k > 0 ) {
			const int32_t j = k - 1;
			const int32_t nofJmers = (int32_t) Math::pow( num_sym, j+1 );
			for(int32_t p = 0; p < num_feat; ++p ) {
				const int32_t offsetJ = nofJmers * p;
				const int32_t offsetJ1 = nofJmers * (p+1);
				const int32_t offsetK = nofKmers * p;
				int32_t y;
				int32_t sym;
				for( y = 0; y < nofJmers; ++y ) {
					for( sym = 0; sym < num_sym; ++sym ) {
						const int32_t y_sym = num_sym*y + sym;
						const int32_t sym_y = nofJmers*sym + y;
						ASSERT(0<=y_sym && y_sym<nofKmers)
						ASSERT(0<=sym_y && sym_y<nofKmers)
						C[k][ y_sym + offsetK ] += L[j][ y + offsetJ ];
						if( p < num_feat-1 ) {
							C[k][ sym_y + offsetK ] += R[j][ y + offsetJ1 ];
						}
					}
				}
			}
		}
		//   if( k > 1 )
		//     j = k-1
		//     for all positions p
		//       for all j-mers y
		//          for n in {A,C,G,T}
		//            C_k[ p, [y,n] ] += L_j[ p, y ]
		//            C_k[ p, [n,y] ] += R_j[ p+1, y ]
		//          end;
		//       end;
		//     end;
		//   end;
	}

	// === return a vector
	num_feat=1;
	num_sym = bigtabSize;
	// --- clean up
	SG_FREE(nofsKmers);
	SG_FREE(margFactors);
	SG_FREE(substrs);
	SG_FREE(x);
	SG_FREE(C);
	for( k = 0; k < max_degree; ++k ) {
		SG_FREE(L[k]);
		SG_FREE(R[k]);
	}
	SG_FREE(L);
	SG_FREE(R);
	return result;
}

char* WeightedDegreePositionStringKernel::compute_consensus(
	int32_t &num_feat, int32_t num_suppvec, int32_t* IDX, float64_t* alphas)
{
	//only works for order <= 32
	ASSERT(degree<=32)
	ASSERT(!tries->get_use_compact_terminal_nodes())
	num_feat=std::static_pointer_cast<StringFeatures<char>>(rhs)->get_max_vector_length();
	ASSERT(num_feat>0)
	auto alphabet = std::static_pointer_cast<StringFeatures<char>>(rhs)->get_alphabet();
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	//consensus
	char* result=SG_MALLOC(char, num_feat);

	//backtracking and scoring table
	int32_t num_tables=Math::max(1,num_feat-degree+1);
	std::vector<ConsensusEntry>** table=SG_MALLOC(std::vector<ConsensusEntry>*, num_tables);

	for (int32_t i=0; i<num_tables; i++)
	{
		table[i]=new std::vector<ConsensusEntry>();
		table[i]->reserve(num_suppvec/10);
	}

	//compute consensus via dynamic programming
	for (auto i : SG_PROGRESS(range(num_tables)))
	{
		bool cumulative=false;

		if (i<num_tables-1)
			init_optimization(num_suppvec, IDX, alphas, i);
		else
		{
			init_optimization(num_suppvec, IDX, alphas, i, num_feat-1);
			cumulative=true;
		}

		if (i==0)
			tries->fill_backtracking_table(i, NULL, table[i], cumulative, weights.vector);
		else
			tries->fill_backtracking_table(i, table[i-1], table[i], cumulative, weights.vector);
	}


	//int32_t n=table[0]->size();

	//for (int32_t i=0; i<n; i++)
	//{
	//	ConsensusEntry e= table[0]->at(i);
	//	SG_PRint32_t("first: str:0%0llx sc:{} bt:{}\n",e.string,e.score,e.bt);
	//}

	//n=table[num_tables-1]->size();
	//for (int32_t i=0; i<n; i++)
	//{
	//	ConsensusEntry e= table[num_tables-1]->at(i);
	//	SG_PRint32_t("last: str:0%0llx sc:{} bt:{}\n",e.string,e.score,e.bt);
	//}
	//n=table[num_tables-2]->size();
	//for (int32_t i=0; i<n; i++)
	//{
	//	ConsensusEntry e= table[num_tables-2]->at(i);
	//	io::print("second last: str:0%0llx sc:{} bt:{}\n",e.string,e.score,e.bt);
	//}

	const char* acgt="ACGT";

	//backtracking start
	int32_t max_idx=-1;
	float32_t max_score=0;
	int32_t num_elements=table[num_tables-1]->size();

	for (int32_t i=0; i<num_elements; i++)
	{
		float64_t sc=table[num_tables-1]->at(i).score;
		if (sc>max_score || max_idx==-1)
		{
			max_idx=i;
			max_score=sc;
		}
	}
	uint64_t endstr=table[num_tables-1]->at(max_idx).string;

	io::info("max_idx:{} num_el:{} num_feat:{} num_tables:{} max_score:{}", max_idx, num_elements, num_feat, num_tables, max_score);

	for (int32_t i=0; i<degree; i++)
		result[num_feat-1-i]=acgt[(endstr >> (2*i)) & 3];

	if (num_tables>1)
	{
		for (int32_t i=num_tables-1; i>=0; i--)
		{
			//io::print("max_idx: {}, i:{}\n", max_idx, i);
			result[i]=acgt[table[i]->at(max_idx).string >> (2*(degree-1)) & 3];
			max_idx=table[i]->at(max_idx).bt;
		}
	}

	//for (int32_t t=0; t<num_tables; t++)
	//{
	//	n=table[t]->size();
	//	for (int32_t i=0; i<n; i++)
	//	{
	//		ConsensusEntry e= table[t]->at(i);
	//		io::print("table[{},{}]: str:0%0llx sc:{:+f} bt:{}\n",t,i, e.string,e.score,e.bt);
	//	}
	//}

	for (int32_t i=0; i<num_tables; i++)
		delete table[i];

	SG_FREE(table);
	return result;
}


float64_t* WeightedDegreePositionStringKernel::extract_w(
	int32_t max_degree, int32_t& num_feat, int32_t& num_sym,
	float64_t* w_result, int32_t num_suppvec, int32_t* IDX, float64_t* alphas)
{
  delete_optimization();
  use_poim_tries=true;
  poim_tries->delete_trees(false);

  // === check
  num_feat=std::static_pointer_cast<StringFeatures<char>>(rhs)->get_max_vector_length();
  ASSERT(num_feat>0)
  auto alphabet = std::static_pointer_cast<StringFeatures<char>>(rhs)->get_alphabet();
  ASSERT(alphabet->get_alphabet()==DNA)
  ASSERT(max_degree>0)

  // === general variables
  static const int32_t NUM_SYMS = poim_tries->NUM_SYMS;
  const int32_t seqLen = num_feat;
  float64_t** subs;
  int32_t i;
  int32_t k;
  //int32_t y;

  // === init tables "subs" for substring scores / POIMs
  // --- compute table sizes
  int32_t* offsets;
  int32_t offset;
  offsets = SG_MALLOC(int32_t,  max_degree );
  offset = 0;
  for( k = 0; k < max_degree; ++k ) {
    offsets[k] = offset;
    const int32_t nofsKmers = (int32_t) Math::pow( NUM_SYMS, k+1 );
    const int32_t tabSize = nofsKmers * seqLen;
    offset += tabSize;
  }
  // --- allocate memory
  const int32_t bigTabSize = offset;
  w_result=SG_MALLOC(float64_t, bigTabSize);
  for (i=0; i<bigTabSize; ++i)
    w_result[i]=0;

  // --- set pointers for tables
  subs = SG_MALLOC(float64_t*,  max_degree );
  ASSERT( subs != NULL )
  for( k = 0; k < max_degree; ++k ) {
    subs[k] = &w_result[ offsets[k] ];
  }
  SG_FREE(offsets);

  // === init trees; extract "w"
  init_optimization( num_suppvec, IDX, alphas, -1);
  poim_tries->POIMs_extract_W( subs, max_degree );

  // === clean; return "subs" as vector
  SG_FREE(subs);
  num_feat = 1;
  num_sym = bigTabSize;
  use_poim_tries=false;
  poim_tries->delete_trees(false);
  return w_result;
}

float64_t* WeightedDegreePositionStringKernel::compute_POIM(
	int32_t max_degree, int32_t& num_feat, int32_t& num_sym,
	float64_t* poim_result, int32_t num_suppvec, int32_t* IDX,
	float64_t* alphas, float64_t* distrib )
{
  delete_optimization();
  use_poim_tries=true;
  poim_tries->delete_trees(false);

  // === check
  num_feat=std::static_pointer_cast<StringFeatures<char>>(rhs)->get_max_vector_length();
  ASSERT(num_feat>0)
  auto alphabet = std::static_pointer_cast<StringFeatures<char>>(rhs)->get_alphabet();
  ASSERT(alphabet->get_alphabet()==DNA)
  ASSERT(max_degree!=0)
  ASSERT(distrib)

  // === general variables
  static const int32_t NUM_SYMS = poim_tries->NUM_SYMS;
  const int32_t seqLen = num_feat;
  float64_t** subs;
  int32_t i;
  int32_t k;

  // === DEBUGGING mode
  //
  // Activated if "max_degree" < 0.
  // Allows to output selected partial score.
  //
  // |max_degree| mod 4
  //   0: substring
  //   1: superstring
  //   2: left overlap
  //   3: right overlap
  //
  const int32_t debug = ( max_degree < 0 ) ? ( abs(max_degree) % 4 + 1 ) : 0;
  if( debug ) {
    max_degree = abs(max_degree) / 4;
    switch( debug ) {
    case 1: {
      printf( "POIM DEBUGGING: substring only (max order=%d)\n", max_degree );
      break;
    }
    case 2: {
      printf( "POIM DEBUGGING: superstring only (max order=%d)\n", max_degree );
      break;
    }
    case 3: {
      printf( "POIM DEBUGGING: left overlap only (max order=%d)\n", max_degree );
      break;
    }
    case 4: {
      printf( "POIM DEBUGGING: right overlap only (max order=%d)\n", max_degree );
      break;
    }
    default: {
      printf( "POIM DEBUGGING: something is wrong (max order=%d)\n", max_degree );
      ASSERT(0)
      break;
    }
    }
  }

  // --- compute table sizes
  int32_t* offsets;
  int32_t offset;
  offsets = SG_MALLOC(int32_t,  max_degree );
  offset = 0;
  for( k = 0; k < max_degree; ++k ) {
    offsets[k] = offset;
    const int32_t nofsKmers = (int32_t) Math::pow( NUM_SYMS, k+1 );
    const int32_t tabSize = nofsKmers * seqLen;
    offset += tabSize;
  }
  // --- allocate memory
  const int32_t bigTabSize=offset;
  poim_result=SG_MALLOC(float64_t, bigTabSize);
  for (i=0; i<bigTabSize; ++i )
    poim_result[i]=0;

  // --- set pointers for tables
  subs=SG_MALLOC(float64_t*, max_degree);
  for (k=0; k<max_degree; ++k)
    subs[k]=&poim_result[offsets[k]];

  SG_FREE(offsets);

  // === init trees; precalc S, L and R
  init_optimization( num_suppvec, IDX, alphas, -1);
  poim_tries->POIMs_precalc_SLR( distrib );

  // === compute substring scores
  if( debug==0 || debug==1 ) {
    poim_tries->POIMs_extract_W( subs, max_degree );
    for( k = 1; k < max_degree; ++k ) {
      const int32_t nofKmers2 = ( k > 1 ) ? (int32_t) Math::pow(NUM_SYMS,k-1) : 0;
      const int32_t nofKmers1 = (int32_t) Math::pow( NUM_SYMS, k );
      const int32_t nofKmers0 = nofKmers1 * NUM_SYMS;
      for( i = 0; i < seqLen; ++i ) {
	float64_t* const subs_k2i1 = ( k>1 && i<seqLen-1 ) ? &subs[k-2][(i+1)*nofKmers2] : NULL;
	float64_t* const subs_k1i1 = ( i < seqLen-1 ) ? &subs[k-1][(i+1)*nofKmers1] : NULL;
	float64_t* const subs_k1i0 = & subs[ k-1 ][ i*nofKmers1 ];
	float64_t* const subs_k0i  = & subs[ k-0 ][ i*nofKmers0 ];
	int32_t y0;
	for( y0 = 0; y0 < nofKmers0; ++y0 ) {
	  const int32_t y1l = y0 / NUM_SYMS;
	  const int32_t y1r = y0 % nofKmers1;
	  const int32_t y2 = y1r / NUM_SYMS;
	  subs_k0i[ y0 ] += subs_k1i0[ y1l ];
	  if( i < seqLen-1 ) {
	    subs_k0i[ y0 ] += subs_k1i1[ y1r ];
	    if( k > 1 ) {
	      subs_k0i[ y0 ] -= subs_k2i1[ y2 ];
	    }
	  }
	}
      }
    }
  }

  // === compute POIMs
  poim_tries->POIMs_add_SLR( subs, max_degree, debug );

  // === clean; return "subs" as vector
  SG_FREE(subs);
  num_feat = 1;
  num_sym = bigTabSize;

  use_poim_tries=false;
  poim_tries->delete_trees(false);

  return poim_result;
}


void WeightedDegreePositionStringKernel::prepare_POIM2(SGMatrix<float64_t> distrib)
{
	m_poim_distrib=distrib.clone();
	m_poim_num_sym=m_poim_distrib.num_cols;
	m_poim_num_feat=m_poim_distrib.num_rows;
}

void WeightedDegreePositionStringKernel::compute_POIM2(
	int32_t max_degree, const std::shared_ptr<SVM>& svm)
{
	ASSERT(svm)
	int32_t num_suppvec=svm->get_num_support_vectors();
	int32_t* sv_idx=SG_MALLOC(int32_t, num_suppvec);
	float64_t* sv_weight=SG_MALLOC(float64_t, num_suppvec);

	for (int32_t i=0; i<num_suppvec; i++)
	{
		sv_idx[i]=svm->get_support_vector(i);
		sv_weight[i]=svm->get_alpha(i);
	}

	if ((max_degree < 1) || (max_degree > 12))
	{
		//io::warn("max_degree out of range 1..12 ({}).", max_degree);
		io::warn("max_degree out of range 1..12 ({}). setting to 1.", max_degree);
		max_degree=1;
	}

	int32_t num_feat = m_poim_num_feat;
	int32_t num_sym = m_poim_num_sym;

	auto* poim = compute_POIM(max_degree, num_feat, num_sym, NULL,	num_suppvec, sv_idx,
						  sv_weight, m_poim_distrib.matrix);
	ASSERT(num_feat==1)
	m_poim = SGVector<float64_t>(poim, num_sym);

	m_poim_result_len=num_sym;

	SG_FREE(sv_weight);
	SG_FREE(sv_idx);
}

SGVector<float64_t> WeightedDegreePositionStringKernel::get_POIM2()
{
	SGVector<float64_t> poim(m_poim, m_poim_result_len, false);
	return poim;
}

void WeightedDegreePositionStringKernel::cleanup_POIM2()
{
	m_poim_num_sym=0 ;
	m_poim_num_sym=0 ;
	m_poim_result_len=0 ;
}

void WeightedDegreePositionStringKernel::load_serializable_post() noexcept(false)
{
	Kernel::load_serializable_post();

	tries=std::make_unique<CTrie<DNATrie>>(degree);
	poim_tries=std::make_unique<CTrie<POIMTrie>>(degree);

	if (weights.size() > 0)
		init_block_weights();
}

void WeightedDegreePositionStringKernel::init()
{
	weights_length = 0;
	weights_degree = 0;
	position_weights_len = 0;

	position_weights_lhs_len = 0;
	position_weights_rhs_len = 0;

	mkl_stepsize = 1;
	degree = 1;
	length = 0;

	max_shift = 0;
	max_mismatch = 0;
	seq_length = 0;
	shift_len = 0;

	block_computation = true;
	type = E_EXTERNAL;
	which_degree = -1;
	tries = std::make_unique<CTrie<DNATrie>>(1);
	poim_tries = std::make_unique<CTrie<POIMTrie>>(1);

	tree_initialized = false;
	use_poim_tries = false;

	m_poim_num_sym = 0;
	m_poim_num_feat = 0;
	m_poim_result_len = 0;

	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;

	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());

	SG_ADD(&weights, "weights", "weights")
	add_callback_function("weights", [this](){
		set_weights(weights);
	});

	SG_ADD(&position_weights, "position_weights", "Weights per position");
	add_callback_function("position_weights", [this](){
		set_position_weights(position_weights);
	});

	SG_ADD(&position_weights_lhs, "position_weights_lhs", "Weights per position left-hand side");
	add_callback_function("position_weights_lhs", [this](){
		set_position_weights_lhs(position_weights_lhs);
	});

	SG_ADD(&position_weights_rhs, "position_weights_rhs", "Weights per position right-hand side");
	add_callback_function("position_weights_rhs", [this](){
		set_position_weights_rhs(position_weights_rhs);
	});

	SG_ADD(&shift, "shifts", "shifts");
	add_callback_function("shifts", [this](){
		set_shifts(shift);
	});

	SG_ADD(
	    &max_shift, "max_shift", "Maximal shift.", ParameterProperties::HYPER);
	SG_ADD(
	    &mkl_stepsize, "mkl_stepsize", "MKL step size.",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &degree, "degree", "Order of WD kernel.", ParameterProperties::HYPER);
	add_callback_function("degree", [this](){
		tries = std::make_unique<CTrie<DNATrie>>(degree);
		poim_tries = std::make_unique<CTrie<POIMTrie>>(degree);
	});
	SG_ADD(
	    &max_mismatch, "max_mismatch", "Number of allowed mismatches.",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &block_computation, "block_computation",
	    "If block computation shall be used.");
	SG_ADD(
	    &which_degree, "which_degree",
	    "The selected degree. All degrees are used by default (for value -1).",
	    ParameterProperties::HYPER);
	SG_ADD_OPTIONS(
	    (machine_int_t*)&type, "type", "WeightedDegree kernel type.",
	    ParameterProperties::HYPER,
	    SG_OPTIONS(
	        E_WD, E_EXTERNAL, E_BLOCK_CONST, E_BLOCK_LINEAR, E_BLOCK_SQPOLY,
	        E_BLOCK_CUBICPOLY, E_BLOCK_EXP, E_BLOCK_LOG));
}
