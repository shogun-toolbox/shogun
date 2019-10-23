/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Heiko Strathmann, Bjoern Esser, Viktor Gal
 */

#include <shogun/lib/common.h>

#ifdef HAVE_COLPACK
#include <ColPack/ColPackHeaders.h>

#include <vector>
#include <string>
#include <cstring>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/ProbingSampler.h>
#include <shogun/mathematics/NormalDistribution.h>

using namespace Eigen;
using namespace ColPack;

namespace shogun
{

ProbingSampler::ProbingSampler() : RandomMixin<TraceSampler>()
{
	init();
}

ProbingSampler::ProbingSampler(
	const std::shared_ptr<SparseMatrixOperator<float64_t>>& matrix_operator, int64_t power,
	EOrderingVariant ordering, EColoringVariant coloring)
	: RandomMixin<TraceSampler>(matrix_operator->get_dimension())
{
	init();

	m_power=power;
	m_matrix_operator=matrix_operator;
	m_ordering=ordering;
	m_coloring=coloring;
}

void ProbingSampler::init()
{
	m_matrix_operator=NULL;
	m_power=1;
	m_ordering=NATURAL;
	m_coloring=DISTANCE_TWO;
	m_is_precomputed=false;

/*
	SG_ADD(&m_coloring_vector, "coloring_vector", "the coloring vector generated"
		" from coloring");

	SG_ADD(&m_power, "matrix_power", "power of the sparse-matrix for coloring");

	SG_ADD(&m_is_precomputed, "is_precomputed",
		"flag that is true if already precomputed");

	SG_ADD((std::shared_ptr<SGObject>*)&m_matrix_operator, "matrix_operator",
		"the sparse-matrix linear opeator for coloring");
		*/
}

ProbingSampler::~ProbingSampler()
{
}

void ProbingSampler::set_coloring_vector(SGVector<int32_t> coloring_vector)
{
	m_coloring_vector=coloring_vector;
	m_is_precomputed=true;
}

SGVector<int32_t> ProbingSampler::get_coloring_vector() const
{
	return m_coloring_vector;
}

void ProbingSampler::precompute()
{
	SG_TRACE("Entering");

	// if already precomputed, nothing to do
	if (m_is_precomputed)
	{
		SG_DEBUG("Coloring vector already computed! Exiting!");
		return;
	}

	// do coloring things here and save the coloring vector
	SparsityStructure* sp_str=m_matrix_operator->get_sparsity_structure(m_power);

	GraphColoringInterface* Color
		=new GraphColoringInterface(SRC_MEM_ADOLC, sp_str->m_ptr, sp_str->m_num_rows);

	std::string str_ordering;
	switch(m_ordering)
	{
	case NATURAL:
		str_ordering="NATURAL";
		break;
	case LARGEST_FIRST:
		str_ordering="LARGEST_FIRST";
		break;
	case DYNAMIC_LARGEST_FIRST:
		str_ordering="DYNAMIC_LARGEST_FIRST";
		break;
	case DISTANCE_TWO_LARGEST_FIRST:
		str_ordering="DISTANCE_TWO_LARGEST_FIRST";
		break;
	case SMALLEST_LAST:
		str_ordering="SMALLEST_LAST";
		break;
	case DISTANCE_TWO_SMALLEST_LAST:
		str_ordering="DISTANCE_TWO_SMALLEST_LAST";
		break;
	case INCIDENCE_DEGREE:
		str_ordering="INCIDENCE_DEGREE";
		break;
	case DISTANCE_TWO_INCIDENCE_DEGREE:
		str_ordering="DISTANCE_TWO_INCIDENCE_DEGREE";
		break;
	case RANDOM:
		str_ordering="RANDOM";
		break;
	}

	std::string str_coloring;
	switch(m_coloring)
	{
	case DISTANCE_ONE:
		str_coloring="DISTANCE_ONE";
		break;
	case ACYCLIC:
		str_coloring="ACYCLIC";
		break;
	case ACYCLIC_FOR_INDIRECT_RECOVERY:
		str_coloring="ACYCLIC_FOR_INDIRECT_RECOVERY";
		break;
	case STAR:
		str_coloring="STAR";
		break;
	case RESTRICTED_STAR:
		str_coloring="RESTRICTED_STAR";
		break;
	case DISTANCE_TWO:
		str_coloring="DISTANCE_TWO";
		break;
	}

	Color->Coloring(str_ordering, str_coloring);

	std::vector<int32_t> vi_VertexColors;
	Color->GetVertexColors(vi_VertexColors);

	require(vi_VertexColors.size()==static_cast<uint32_t>(m_dimension),
		"dimension mismatch, {} vs {}!", vi_VertexColors.size(), m_dimension);

	m_coloring_vector=SGVector<int32_t>(vi_VertexColors.size());

	for (std::vector<int32_t>::iterator it=vi_VertexColors.begin();
		it!=vi_VertexColors.end(); it++)
	{
		index_t i=static_cast<index_t>(std::distance(vi_VertexColors.begin(), it));
		m_coloring_vector[i]=*it;
	}

	Map<VectorXi> colors(m_coloring_vector.vector, m_coloring_vector.vlen);
	m_num_samples=colors.maxCoeff()+1;
	SG_DEBUG("Using {} samples (aka colours) for probing trace sampler",
			m_num_samples);

	delete sp_str;
	delete Color;

	// set the precomputed flag true
	m_is_precomputed=true;

	SG_TRACE("Leaving");
}

SGVector<float64_t> ProbingSampler::sample(index_t idx) const
{
	require(idx<m_num_samples, "Given index ({}) must be smaller than "
			"number of samples to draw ({})", idx, m_num_samples);

	SGVector<float64_t> s(m_dimension);
	s.set_const(0.0);

	NormalDistribution<float64_t> normal_dist;
	for (index_t i=0; i<m_dimension; ++i)
	{
		if (m_coloring_vector[i]==idx)
		{
			float64_t x=normal_dist(m_prng);
			s[i]=(x>0)-(x<0);
		}
	}

	return s;
}

}

#endif // HAVE_COLPACK
