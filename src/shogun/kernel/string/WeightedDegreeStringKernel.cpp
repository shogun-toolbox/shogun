/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Soeren Sonnenburg, Sergey Lisitsyn, Bjoern Esser,
 *          Viktor Gal
 */

#include <shogun/base/Parallel.h>
#include <shogun/base/progress.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Trie.h>
#include <shogun/lib/common.h>

#include <shogun/kernel/string/WeightedDegreeStringKernel.h>
#include <shogun/kernel/normalizer/FirstElementKernelNormalizer.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>

#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct S_THREAD_PARAM_WD
{

	int32_t* vec;
	float64_t* result;
	float64_t* weights;
	WeightedDegreeStringKernel* kernel;
	CTrie<DNATrie>* tries;
	float64_t factor;
	int32_t j;
	int32_t start;
	int32_t end;
	int32_t length;
	int32_t* vec_idx;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

WeightedDegreeStringKernel::WeightedDegreeStringKernel ()
: StringKernel<char>()
{
	init();
}


WeightedDegreeStringKernel::WeightedDegreeStringKernel (
	int32_t d, EWDKernType t)
: StringKernel<char>()
{
	init();

	degree=d;
	type=t;
}

WeightedDegreeStringKernel::WeightedDegreeStringKernel(SGVector<float64_t> w)
: StringKernel<char>(10)
{
	init();

	type=E_EXTERNAL;
	degree=w.vlen;

	weights=SG_MALLOC(float64_t, degree*(1+max_mismatch));
	weights_degree=degree;
	weights_length=(1+max_mismatch);

	for (int32_t i=0; i<degree*(1+max_mismatch); i++)
		weights[i]=w.vector[i];
}

WeightedDegreeStringKernel::WeightedDegreeStringKernel(
	std::shared_ptr<StringFeatures<char>> l, std::shared_ptr<StringFeatures<char>> r, int32_t d)
: StringKernel<char>(10)
{
	init();
	degree=d;
	type=E_WD;
	set_wd_weights_by_type(type);
	set_normalizer(std::make_shared<FirstElementKernelNormalizer>());
	init(l, r);
}

WeightedDegreeStringKernel::~WeightedDegreeStringKernel()
{
	cleanup();

	SG_FREE(weights);
	weights=NULL;
	weights_degree=0;
	weights_length=0;

	SG_FREE(block_weights);
	block_weights=NULL;

	SG_FREE(position_weights);
	position_weights=NULL;

	SG_FREE(weights_buffer);
	weights_buffer=NULL;
}


void WeightedDegreeStringKernel::remove_lhs()
{
	SG_DEBUG("deleting CWeightedDegreeStringKernel optimization")
	delete_optimization();

	if (tries!=NULL)
		tries->destroy();

	Kernel::remove_lhs();
}

void WeightedDegreeStringKernel::create_empty_tries()
{
	ASSERT(lhs)

	seq_length=lhs->as<StringFeatures<char>>()->get_max_vector_length();

	if (tries!=NULL)
	{
		tries->destroy() ;
		tries->create(seq_length, max_mismatch==0) ;
	}
}

bool WeightedDegreeStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	if (type != E_EXTERNAL)
		set_wd_weights_by_type(type);
	
	int32_t lhs_changed=(lhs!=l);
	int32_t rhs_changed=(rhs!=r);

	StringKernel<char>::init(l,r);

	SG_DEBUG("lhs_changed: {}", lhs_changed)
	SG_DEBUG("rhs_changed: {}", rhs_changed)

	auto sf_l=l->as<StringFeatures<char>>();
	auto sf_r=r->as<StringFeatures<char>>();

	int32_t len=sf_l->get_max_vector_length();
	if (lhs_changed && !sf_l->have_same_length(len))
		error("All strings in WD kernel must have same length (lhs wrong)!");

	if (rhs_changed && !sf_r->have_same_length(len))
		error("All strings in WD kernel must have same length (rhs wrong)!");


	alphabet=sf_l->get_alphabet();
	auto ralphabet=sf_r->get_alphabet();

	if (!((alphabet->get_alphabet()==DNA) || (alphabet->get_alphabet()==RNA)))
		properties &= ((uint64_t) (-1)) ^ (KP_LINADD | KP_BATCHEVALUATION);

	ASSERT(ralphabet->get_alphabet()==alphabet->get_alphabet())


	if (tries!=NULL) {
		tries->delete_trees(max_mismatch==0);

	}
	tries=std::make_shared<CTrie<DNATrie>>(degree, max_mismatch==0);
	create_empty_tries();

	init_block_weights();

	return init_normalizer();
}

void WeightedDegreeStringKernel::cleanup()
{
	SG_DEBUG("deleting CWeightedDegreeStringKernel optimization")
	delete_optimization();

	SG_FREE(block_weights);
	block_weights=NULL;

	if (tries!=NULL)
	{
		tries->destroy();

		tries=NULL;
	}

	seq_length=0;
	tree_initialized = false;


	alphabet=NULL;

	Kernel::cleanup();
}

bool WeightedDegreeStringKernel::init_optimization(int32_t count, int32_t* IDX, float64_t* alphas, int32_t tree_num)
{
	if (tree_num<0)
		SG_DEBUG("deleting CWeightedDegreeStringKernel optimization")

	delete_optimization();

	if (tree_num<0)
		SG_DEBUG("initializing CWeightedDegreeStringKernel optimization")

	for (auto i : SG_PROGRESS(range(count)))
	{
		if (tree_num<0)
		{

			if (max_mismatch==0)
				add_example_to_tree(IDX[i], alphas[i]) ;
			else
				add_example_to_tree_mismatch(IDX[i], alphas[i]) ;

			//SG_DEBUG("number of used trie nodes: {}", tries.get_num_used_nodes())
		}
		else
		{
			if (max_mismatch==0)
				add_example_to_single_tree(IDX[i], alphas[i], tree_num) ;
			else
				add_example_to_single_tree_mismatch(IDX[i], alphas[i], tree_num) ;
		}
	}

	//tries.compact_nodes(NO_CHILD, 0, weights) ;

	set_is_initialized(true) ;
	return true ;
}

bool WeightedDegreeStringKernel::delete_optimization()
{
	if (get_is_initialized())
	{
		if (tries!=NULL)
			tries->delete_trees(max_mismatch==0);
		set_is_initialized(false);
		return true;
	}

	return false;
}


float64_t WeightedDegreeStringKernel::compute_with_mismatch(
	char* avec, int32_t alen, char* bvec, int32_t blen)
{
	float64_t sum = 0.0;

	for (int32_t i=0; i<alen; i++)
	{
		float64_t sumi = 0.0;
		int32_t mismatches=0;

		for (int32_t j=0; (i+j<alen) && (j<degree); j++)
		{
			if (avec[i+j]!=bvec[i+j])
			{
				mismatches++ ;
				if (mismatches>max_mismatch)
					break ;
			} ;
			sumi += weights[j+degree*mismatches];
		}
		if (position_weights!=NULL)
			sum+=position_weights[i]*sumi ;
		else
			sum+=sumi ;
	}
	return sum ;
}

float64_t WeightedDegreeStringKernel::compute_using_block(
	char* avec, int32_t alen, char* bvec, int32_t blen)
{
	ASSERT(alen==blen)

	float64_t sum=0;
	int32_t match_len=-1;

	for (int32_t i=0; i<alen; i++)
	{
		if (avec[i]==bvec[i])
			match_len++;
		else
		{
			if (match_len>=0)
				sum+=block_weights[match_len];
			match_len=-1;
		}
	}

	if (match_len>=0)
		sum+=block_weights[match_len];

	return sum;
}

float64_t WeightedDegreeStringKernel::compute_without_mismatch(
	char* avec, int32_t alen, char* bvec, int32_t blen)
{
	float64_t sum = 0.0;

	for (int32_t i=0; i<alen; i++)
	{
		float64_t sumi = 0.0;

		for (int32_t j=0; (i+j<alen) && (j<degree); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break ;
			sumi += weights[j];
		}
		if (position_weights!=NULL)
			sum+=position_weights[i]*sumi ;
		else
			sum+=sumi ;
	}
	return sum ;
}

float64_t WeightedDegreeStringKernel::compute_without_mismatch_matrix(
	char* avec, int32_t alen, char* bvec, int32_t blen)
{
	float64_t sum = 0.0;

	for (int32_t i=0; i<alen; i++)
	{
		float64_t sumi=0.0;
		for (int32_t j=0; (i+j<alen) && (j<degree); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break;
			sumi += weights[i*degree+j];
		}
		if (position_weights!=NULL)
			sum += position_weights[i]*sumi ;
		else
			sum += sumi ;
	}

	return sum ;
}


float64_t WeightedDegreeStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;
	char* avec=lhs->as<StringFeatures<char>>()->get_feature_vector(idx_a, alen, free_avec);
	char* bvec=rhs->as<StringFeatures<char>>()->get_feature_vector(idx_b, blen, free_bvec);
	float64_t result=0;

	if (max_mismatch==0 && length==0 && block_computation)
		result=compute_using_block(avec, alen, bvec, blen);
	else
	{
		if (max_mismatch>0)
			result=compute_with_mismatch(avec, alen, bvec, blen);
		else if (length==0)
			result=compute_without_mismatch(avec, alen, bvec, blen);
		else
			result=compute_without_mismatch_matrix(avec, alen, bvec, blen);
	}
	lhs->as<StringFeatures<char>>()->free_feature_vector(avec, idx_a, free_avec);
	rhs->as<StringFeatures<char>>()->free_feature_vector(bvec, idx_b, free_bvec);

	return result;
}


void WeightedDegreeStringKernel::add_example_to_tree(
	int32_t idx, float64_t alpha)
{
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	int32_t len=0;
	bool free_vec;
	char* char_vec=lhs->as<StringFeatures<char>>()->get_feature_vector(idx, len, free_vec);
	ASSERT(max_mismatch==0)
	int32_t *vec=SG_MALLOC(int32_t, len);

	for (int32_t i=0; i<len; i++)
		vec[i]=alphabet->remap_to_bin(char_vec[i]);
	lhs->as<StringFeatures<char>>()->free_feature_vector(char_vec, idx, free_vec);

	if (length == 0 || max_mismatch > 0)
	{
		for (int32_t i=0; i<len; i++)
		{
			float64_t alpha_pw=alpha;
			/*if (position_weights!=NULL)
			  alpha_pw *= position_weights[i] ;*/
			if (alpha_pw==0.0)
				continue;
			ASSERT(tries)
			tries->add_to_trie(i, 0, vec, normalizer->normalize_lhs(alpha_pw, idx), weights, (length!=0));
		}
	}
	else
	{
		for (int32_t i=0; i<len; i++)
		{
			float64_t alpha_pw=alpha;
			/*if (position_weights!=NULL)
			  alpha_pw = alpha*position_weights[i] ;*/
			if (alpha_pw==0.0)
				continue ;
			ASSERT(tries)
			tries->add_to_trie(i, 0, vec, normalizer->normalize_lhs(alpha_pw, idx), weights, (length!=0));
		}
	}
	SG_FREE(vec);
	tree_initialized=true ;
}

void WeightedDegreeStringKernel::add_example_to_single_tree(
	int32_t idx, float64_t alpha, int32_t tree_num)
{
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	int32_t len;
	bool free_vec;
	char* char_vec=lhs->as<StringFeatures<char>>()->get_feature_vector(idx, len, free_vec);
	ASSERT(max_mismatch==0)
	int32_t *vec = SG_MALLOC(int32_t, len);

	for (int32_t i=tree_num; i<tree_num+degree && i<len; i++)
		vec[i]=alphabet->remap_to_bin(char_vec[i]);
	lhs->as<StringFeatures<char>>()->free_feature_vector(char_vec, idx, free_vec);


	ASSERT(tries)
	if (alpha!=0.0)
		tries->add_to_trie(tree_num, 0, vec, normalizer->normalize_lhs(alpha, idx), weights, (length!=0));

	SG_FREE(vec);
	tree_initialized=true ;
}

void WeightedDegreeStringKernel::add_example_to_tree_mismatch(int32_t idx, float64_t alpha)
{
	ASSERT(tries)
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	int32_t len ;
	bool free_vec;
	char* char_vec=lhs->as<StringFeatures<char>>()->get_feature_vector(idx, len, free_vec);

	int32_t *vec = SG_MALLOC(int32_t, len);

	for (int32_t i=0; i<len; i++)
		vec[i]=alphabet->remap_to_bin(char_vec[i]);
	lhs->as<StringFeatures<char>>()->free_feature_vector(char_vec, idx, free_vec);

	for (int32_t i=0; i<len; i++)
	{
		if (alpha!=0.0)
			tries->add_example_to_tree_mismatch_recursion(NO_CHILD, i, normalizer->normalize_lhs(alpha, idx), &vec[i], len-i, 0, 0, max_mismatch, weights);
	}

	SG_FREE(vec);
	tree_initialized=true ;
}

void WeightedDegreeStringKernel::add_example_to_single_tree_mismatch(
	int32_t idx, float64_t alpha, int32_t tree_num)
{
	ASSERT(tries)
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	int32_t len=0;
	bool free_vec;
	char* char_vec=lhs->as<StringFeatures<char>>()->get_feature_vector(idx, len, free_vec);
	int32_t *vec=SG_MALLOC(int32_t, len);

	for (int32_t i=tree_num; i<len && i<tree_num+degree; i++)
		vec[i]=alphabet->remap_to_bin(char_vec[i]);
	lhs->as<StringFeatures<char>>()->free_feature_vector(char_vec, idx, free_vec);

	if (alpha!=0.0)
	{
		tries->add_example_to_tree_mismatch_recursion(
			NO_CHILD, tree_num, normalizer->normalize_lhs(alpha, idx), &vec[tree_num], len-tree_num,
			0, 0, max_mismatch, weights);
	}

	SG_FREE(vec);
	tree_initialized=true;
}


float64_t WeightedDegreeStringKernel::compute_by_tree(int32_t idx)
{
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	int32_t len=0;
	bool free_vec;
	char* char_vec=rhs->as<StringFeatures<char>>()->get_feature_vector(idx, len, free_vec);
	ASSERT(char_vec && len>0)
	int32_t *vec=SG_MALLOC(int32_t, len);

	for (int32_t i=0; i<len; i++)
		vec[i]=alphabet->remap_to_bin(char_vec[i]);
	lhs->as<StringFeatures<char>>()->free_feature_vector(char_vec, idx, free_vec);

	float64_t sum=0;
	ASSERT(tries)
	for (int32_t i=0; i<len; i++)
		sum+=tries->compute_by_tree_helper(vec, len, i, i, i, weights, (length!=0));

	SG_FREE(vec);
	return normalizer->normalize_rhs(sum, idx);
}

void WeightedDegreeStringKernel::compute_by_tree(
	int32_t idx, float64_t* LevelContrib)
{
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	int32_t len ;
	bool free_vec;
	char* char_vec=rhs->as<StringFeatures<char>>()->get_feature_vector(idx, len, free_vec);

	int32_t *vec = SG_MALLOC(int32_t, len);

	for (int32_t i=0; i<len; i++)
		vec[i]=alphabet->remap_to_bin(char_vec[i]);
	lhs->as<StringFeatures<char>>()->free_feature_vector(char_vec, idx, free_vec);

	ASSERT(tries)
	for (int32_t i=0; i<len; i++)
	{
		tries->compute_by_tree_helper(vec, len, i, i, i, LevelContrib,
				normalizer->normalize_rhs(1.0, idx),
				mkl_stepsize, weights, (length!=0));
	}

	SG_FREE(vec);
}

float64_t *WeightedDegreeStringKernel::compute_abs_weights(int32_t &len)
{
	ASSERT(tries)
	return tries->compute_abs_weights(len);
}

bool WeightedDegreeStringKernel::set_wd_weights_by_type(EWDKernType p_type)
{
	ASSERT(degree>0)
	ASSERT(p_type==E_WD) /// if we know a better weighting later on do a switch

	SG_FREE(weights);
	weights=SG_MALLOC(float64_t, degree);
	weights_degree=degree;
	weights_length=1;

	if (weights)
	{
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

		if (which_degree>=0)
		{
			ASSERT(which_degree<degree)
			for (i=0; i<degree; i++)
			{
				if (i!=which_degree)
					weights[i]=0;
				else
					weights[i]=1;
			}
		}
		return true;
	}
	else
		return false;
}

bool WeightedDegreeStringKernel::set_weights(SGMatrix<float64_t> new_weights)
{
	float64_t* ws=new_weights.matrix;
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
	int32_t num_weights=weights_degree*weights_length;
	SG_FREE(weights);
	weights=SG_MALLOC(float64_t, num_weights);

	for (int32_t i=0; i<degree*len; i++)
		weights[i]=ws[i];

	return true;
}

bool WeightedDegreeStringKernel::set_position_weights(
	float64_t* pws, int32_t len)
{
	if (len==0)
	{
		SG_FREE(position_weights);
		position_weights=NULL;
		ASSERT(tries)
		tries->set_position_weights(position_weights);
	}

	if (seq_length!=len)
		error("seq_length = {}, position_weights_length={}", seq_length, len);

	SG_FREE(position_weights);
	position_weights=SG_MALLOC(float64_t, len);
	position_weights_len=len;
	ASSERT(tries)
	tries->set_position_weights(position_weights);

	if (position_weights)
	{
		for (int32_t i=0; i<len; i++)
			position_weights[i]=pws[i];
		return true;
	}
	else
		return false;
}

bool WeightedDegreeStringKernel::init_block_weights_from_wd()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, Math::max(seq_length,degree));

	int32_t k;
	float64_t d=degree; // use float to evade rounding errors below

	for (k=0; k<degree; k++)
		block_weights[k]=
			(-Math::pow(k, 3)+(3*d-3)*Math::pow(k, 2)+(9*d-2)*k+6*d)/(3*d*(d+1));
	for (k=degree; k<seq_length; k++)
		block_weights[k]=(-d+3*k+4)/3;

	return true;
}

bool WeightedDegreeStringKernel::init_block_weights_from_wd_external()
{
	ASSERT(weights)
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, Math::max(seq_length,degree));

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
	return true;
}

bool WeightedDegreeStringKernel::init_block_weights_const()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, seq_length);

	for (int32_t i=1; i<seq_length+1 ; i++)
		block_weights[i-1]=1.0/seq_length;
	return true;
}

bool WeightedDegreeStringKernel::init_block_weights_linear()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, seq_length);

	for (int32_t i=1; i<seq_length+1 ; i++)
		block_weights[i-1]=degree*i;

	return true;
}

bool WeightedDegreeStringKernel::init_block_weights_sqpoly()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, seq_length);

	for (int32_t i=1; i<degree+1 ; i++)
		block_weights[i-1]=((float64_t) i)*i;

	for (int32_t i=degree+1; i<seq_length+1 ; i++)
		block_weights[i-1]=i;

	return true;
}

bool WeightedDegreeStringKernel::init_block_weights_cubicpoly()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, seq_length);

	for (int32_t i=1; i<degree+1 ; i++)
		block_weights[i-1]=((float64_t) i)*i*i;

	for (int32_t i=degree+1; i<seq_length+1 ; i++)
		block_weights[i-1]=i;
	return true;
}

bool WeightedDegreeStringKernel::init_block_weights_exp()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, seq_length);

	for (int32_t i=1; i<degree+1 ; i++)
		block_weights[i-1]=exp(((float64_t) i/10.0));

	for (int32_t i=degree+1; i<seq_length+1 ; i++)
		block_weights[i-1]=i;

	return true;
}

bool WeightedDegreeStringKernel::init_block_weights_log()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, seq_length);

	for (int32_t i=1; i<degree+1 ; i++)
		block_weights[i - 1] = Math::pow(std::log((float64_t)i), 2);

	for (int32_t i=degree+1; i<seq_length+1 ; i++)
		block_weights[i - 1] =
		    i - degree + 1 + Math::pow(std::log(degree + 1.0), 2);

	return true;
}

bool WeightedDegreeStringKernel::init_block_weights()
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


void* WeightedDegreeStringKernel::compute_batch_helper(void* p)
{
	S_THREAD_PARAM_WD* params = (S_THREAD_PARAM_WD*) p;
	int32_t j=params->j;
	WeightedDegreeStringKernel* wd=params->kernel;
	CTrie<DNATrie>* tries=params->tries;
	float64_t* weights=params->weights;
	int32_t length=params->length;
	int32_t* vec=params->vec;
	float64_t* result=params->result;
	float64_t factor=params->factor;
	int32_t* vec_idx=params->vec_idx;

	StringFeatures<char>* rhs_feat=(StringFeatures<char>*)(wd->get_rhs().get());
	auto alpha=wd->alphabet;

	for (int32_t i=params->start; i<params->end; i++)
	{
		int32_t len=0;
		bool free_vec;
		char* char_vec=rhs_feat->get_feature_vector(vec_idx[i], len, free_vec);
		for (int32_t k=j; k<Math::min(len,j+wd->get_degree()); k++)
			vec[k]=alpha->remap_to_bin(char_vec[k]);
		rhs_feat->free_feature_vector(char_vec, vec_idx[i], free_vec);

		ASSERT(tries)

		result[i]+=factor*
			wd->normalizer->normalize_rhs(tries->compute_by_tree_helper(vec, len, j, j, j, weights, (length!=0)), vec_idx[i]);
	}



	return NULL;
}

void WeightedDegreeStringKernel::compute_batch(
	int32_t num_vec, int32_t* vec_idx, float64_t* result, int32_t num_suppvec,
	int32_t* IDX, float64_t* alphas, float64_t factor)
{
	ASSERT(tries)
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)
	ASSERT(rhs)
	ASSERT(num_vec<=rhs->get_num_vectors())
	ASSERT(num_vec>0)
	ASSERT(vec_idx)
	ASSERT(result)
	create_empty_tries();

	int32_t num_feat=rhs->as<StringFeatures<char>>()->get_max_vector_length();
	ASSERT(num_feat>0)
	// TODO: port to use OpenMP backend instead of pthread
#ifdef HAVE_PTHREAD
	int32_t num_threads=env()->get_num_threads();
#else
	int32_t num_threads=1;
#endif
	ASSERT(num_threads>0)
	int32_t* vec=SG_MALLOC(int32_t, num_threads*num_feat);
	auto pb = SG_PROGRESS(range(num_feat));

	if (num_threads < 2)
	{
		// TODO: replace with the new signal
		// for (int32_t j=0; j<num_feat && !Signal::cancel_computations(); j++)
		for (int32_t j = 0; j < num_feat; j++)
		{
			init_optimization(num_suppvec, IDX, alphas, j);
			S_THREAD_PARAM_WD params;
			params.vec=vec;
			params.result=result;
			params.weights=weights;
			params.kernel=this;
			params.tries=tries.get();
			params.factor=factor;
			params.j=j;
			params.start=0;
			params.end=num_vec;
			params.length=length;
			params.vec_idx=vec_idx;
			compute_batch_helper((void*) &params);

			pb.print_progress();
		}
		pb.complete();
	}
#ifdef HAVE_PTHREAD
	else
	{
		// TODO: replace with the new signal
		// for (int32_t j=0; j<num_feat && !Signal::cancel_computations(); j++)
		for (int32_t j = 0; j < num_feat; j++)
		{
			init_optimization(num_suppvec, IDX, alphas, j);
			pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
			S_THREAD_PARAM_WD* params = SG_MALLOC(S_THREAD_PARAM_WD, num_threads);
			int32_t step= num_vec/num_threads;
			int32_t t;

			for (t=0; t<num_threads-1; t++)
			{
				params[t].vec=&vec[num_feat*t];
				params[t].result=result;
				params[t].weights=weights;
				params[t].kernel=this;
				params[t].tries=tries.get();
				params[t].factor=factor;
				params[t].j=j;
				params[t].start = t*step;
				params[t].end = (t+1)*step;
				params[t].length=length;
				params[t].vec_idx=vec_idx;
				pthread_create(&threads[t], NULL, WeightedDegreeStringKernel::compute_batch_helper, (void*)&params[t]);
			}
			params[t].vec=&vec[num_feat*t];
			params[t].result=result;
			params[t].weights=weights;
			params[t].kernel=this;
			params[t].tries=tries.get();
			params[t].factor=factor;
			params[t].j=j;
			params[t].start=t*step;
			params[t].end=num_vec;
			params[t].length=length;
			params[t].vec_idx=vec_idx;
			compute_batch_helper((void*) &params[t]);

			for (t=0; t<num_threads-1; t++)
				pthread_join(threads[t], NULL);
			pb.print_progress();

			SG_FREE(params);
			SG_FREE(threads);
		}
		pb.complete();
	}
#endif

	SG_FREE(vec);

	//really also free memory as this can be huge on testing especially when
	//using the combined kernel
	create_empty_tries();
}

bool WeightedDegreeStringKernel::set_max_mismatch(int32_t max)
{
	if (type==E_EXTERNAL && max!=0)
		return false;

	max_mismatch=max;

	if (lhs!=NULL && rhs!=NULL)
		return init(lhs, rhs);
	else
		return true;
}

void WeightedDegreeStringKernel::init()
{
	type = E_WD;

	weights = NULL;
	weights_degree = 0;
	weights_length = 0;

	position_weights = NULL;
	position_weights_len = 0;

	weights_buffer = NULL;
	mkl_stepsize = 1;
	degree = 1;
	length = 0;

	max_mismatch = 0;
	seq_length = 0;

	block_weights = NULL;
	block_computation = true;
	type = E_WD;
	which_degree = -1;
	tries = NULL;

	tree_initialized = false;
	alphabet = NULL;

	lhs = NULL;
	rhs = NULL;

	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;

	set_normalizer(std::make_shared<FirstElementKernelNormalizer>());

	/*m_parameters->add_matrix(
	    &weights, &weights_degree, &weights_length, "weights",
	    "WD Kernel weights.");*/
	watch_param("weights", &weights, &weights_degree, &weights_length);

	/*m_parameters->add_vector(
	    &position_weights, &position_weights_len, "position_weights",
	    "Weights per position.");*/
	watch_param("position_weights", &position_weights, &position_weights_len);

	SG_ADD(
	    &mkl_stepsize, "mkl_stepsize", "MKL step size.",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &degree, "degree", "Order of WD kernel.", ParameterProperties::HYPER);
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
	SG_ADD((std::shared_ptr<SGObject>*)&alphabet, "alphabet", "Alphabet of Features.");
	SG_ADD_OPTIONS(
	    (machine_int_t*)&type, "type", "WeightedDegree kernel type.",
	    ParameterProperties::HYPER,
	    SG_OPTIONS(
	        E_WD, E_EXTERNAL, E_BLOCK_CONST, E_BLOCK_LINEAR, E_BLOCK_SQPOLY,
	        E_BLOCK_CUBICPOLY, E_BLOCK_EXP, E_BLOCK_LOG));
}
