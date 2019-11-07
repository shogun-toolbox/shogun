/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/string/RegulatoryModulesStringKernel.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>

#include <utility>

using namespace shogun;

RegulatoryModulesStringKernel::RegulatoryModulesStringKernel()
: StringKernel<char>(0)
{
	init();
}

RegulatoryModulesStringKernel::RegulatoryModulesStringKernel(
		int32_t size, float64_t w, int32_t d, int32_t s, int32_t wl)
: StringKernel<char>(size)
{
	init();
}

RegulatoryModulesStringKernel::RegulatoryModulesStringKernel(const std::shared_ptr<StringFeatures<char>>& lstr, const std::shared_ptr<StringFeatures<char>>& rstr,
		std::shared_ptr<DenseFeatures<uint16_t>> lpos, std::shared_ptr<DenseFeatures<uint16_t>> rpos,
		float64_t w, int32_t d, int32_t s, int32_t wl, int32_t size)
: StringKernel<char>(size)
{
	init();
	set_motif_positions(std::move(lpos), std::move(rpos));
	init(lstr,rstr);
}

RegulatoryModulesStringKernel::~RegulatoryModulesStringKernel()
{


}

void RegulatoryModulesStringKernel::init()
{
	width=0;
	degree=0;
	shift=0;
	window=0;
	motif_positions_lhs=NULL;
	motif_positions_rhs=NULL;

	SG_ADD(&width, "width", "the width of Gaussian kernel part", ParameterProperties::HYPER);
	SG_ADD(&degree, "degree", "the degree of weighted degree kernel part",
	    ParameterProperties::HYPER);
	SG_ADD(&shift, "shift",
	    "the shift of weighted degree with shifts kernel part", ParameterProperties::HYPER);
	SG_ADD(&window, "window", "the size of window around motifs", ParameterProperties::HYPER);
	SG_ADD((std::shared_ptr<SGObject>*)&motif_positions_lhs, "motif_positions_lhs",
			"the matrix of motif positions from sequences left-hand side");
	SG_ADD((std::shared_ptr<SGObject>*)&motif_positions_rhs, "motif_positions_rhs",
			"the matrix of motif positions from sequences right-hand side");
	SG_ADD(&position_weights, "position_weights", "scaling weights in window");
	SG_ADD(&weights, "weights", "weights of WD kernel");
}

bool RegulatoryModulesStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(motif_positions_lhs)
	ASSERT(motif_positions_rhs)

	if (l->get_num_vectors() != motif_positions_lhs->get_num_vectors())
		error("Number of vectors does not agree (LHS: {}, Motif LHS: {}).",
				l->get_num_vectors(),  motif_positions_lhs->get_num_vectors());
	if (r->get_num_vectors() != motif_positions_rhs->get_num_vectors())
		error("Number of vectors does not agree (RHS: {}, Motif RHS: {}).",
				r->get_num_vectors(), motif_positions_rhs->get_num_vectors());

	set_wd_weights();
	StringKernel<char>::init(l, r);
	return init_normalizer();
}

void RegulatoryModulesStringKernel::set_motif_positions(
		const std::shared_ptr<DenseFeatures<uint16_t>>& positions_lhs, const std::shared_ptr<DenseFeatures<uint16_t>>& positions_rhs)
{
	ASSERT(positions_lhs)
	ASSERT(positions_rhs)


	if (positions_lhs->get_num_features() != positions_rhs->get_num_features())
		error("Number of dimensions does not agree.");

	motif_positions_lhs=positions_lhs;
	motif_positions_rhs=positions_rhs;


}

float64_t RegulatoryModulesStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	ASSERT(motif_positions_lhs)
	ASSERT(motif_positions_rhs)

	bool free_avec, free_bvec;
	int32_t alen=0;
	int32_t blen=0;
	char* avec=std::static_pointer_cast<StringFeatures<char>>(lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec=std::static_pointer_cast<StringFeatures<char>>(rhs)->get_feature_vector(idx_b, blen, free_bvec);

	int32_t alen_pos, blen_pos;
	bool afree_pos, bfree_pos;
	uint16_t* positions_a = motif_positions_lhs->get_feature_vector(idx_a, alen_pos, afree_pos);
	uint16_t* positions_b = motif_positions_rhs->get_feature_vector(idx_b, blen_pos, bfree_pos);
	ASSERT(alen_pos==blen_pos)
	int32_t num_pos=alen_pos;


	float64_t result_rbf=0;
	float64_t result_wds=0;

	for (int32_t p=0; p<num_pos; p++)
	{
		result_rbf+=Math::sq(positions_a[p]-positions_b[p]);

		for (int32_t p2=0; p2<num_pos; p2++) //p+1 and below * 2
			result_rbf+=Math::sq( (positions_a[p]-positions_a[p2]) - (positions_b[p]-positions_b[p2]) );

		int32_t limit = window;
		if (window + positions_a[p] > alen)
			limit = alen - positions_a[p];

		if (window + positions_b[p] > blen)
			limit = Math::min(limit, blen - positions_b[p]);

		result_wds+=compute_wds(&avec[positions_a[p]], &bvec[positions_b[p]],
				limit);
	}

	float64_t result=exp(-result_rbf/width)+result_wds;

	std::static_pointer_cast<StringFeatures<char>>(lhs)->free_feature_vector(avec, idx_a, free_avec);
	std::static_pointer_cast<StringFeatures<char>>(rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	std::static_pointer_cast<DenseFeatures<uint16_t>>(lhs)->free_feature_vector(positions_a, idx_a, afree_pos);
	std::static_pointer_cast<DenseFeatures<uint16_t>>(rhs)->free_feature_vector(positions_b, idx_b, bfree_pos);

	return result;
}

float64_t RegulatoryModulesStringKernel::compute_wds(
	char* avec, char* bvec, int32_t len)
{
	float64_t* max_shift_vec = SG_MALLOC(float64_t, shift);
	float64_t sum0=0 ;
	for (int32_t i=0; i<shift; i++)
		max_shift_vec[i]=0 ;

	// no shift
	for (int32_t i=0; i<len; i++)
	{
		if ((position_weights.vector!=NULL) && (position_weights[i]==0.0))
			continue ;

		float64_t sumi = 0.0 ;
		for (int32_t j=0; (j<degree) && (i+j<len); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break ;
			sumi += weights[j];
		}
		if (position_weights.vector!=NULL)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
	} ;

	for (int32_t i=0; i<len; i++)
	{
		for (int32_t k=1; (k<=shift) && (i+k<len); k++)
		{
			if ((position_weights.vector!=NULL) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;

			float64_t sumi1 = 0.0 ;
			// shift in sequence a
			for (int32_t j=0; (j<degree) && (i+j+k<len); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi1 += weights[j];
			}
			float64_t sumi2 = 0.0 ;
			// shift in sequence b
			for (int32_t j=0; (j<degree) && (i+j+k<len); j++)
			{
				if (avec[i+j]!=bvec[i+j+k])
					break ;
				sumi2 += weights[j];
			}
			if (position_weights.vector!=NULL)
				max_shift_vec[k-1] += position_weights[i]*sumi1 + position_weights[i+k]*sumi2 ;
			else
				max_shift_vec[k-1] += sumi1 + sumi2 ;
		} ;
	}

	float64_t result = sum0 ;
	for (int32_t i=0; i<shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;

	SG_FREE(max_shift_vec);
	return result ;
}

void RegulatoryModulesStringKernel::set_wd_weights()
{
	ASSERT(degree>0)

	weights=SGVector<float64_t>(degree);

	int32_t i;
	float64_t sum=0;
	for (i=0; i<degree; i++)
	{
		weights[i]=degree-i;
		sum+=weights[i];
	}

	for (i=0; i<degree; i++)
		weights[i]/=sum;
}

