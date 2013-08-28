/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/common.h>

#ifdef HAVE_COLPACK
#ifdef HAVE_EIGEN3

#include <vector>
#include <string>
#include <cstring>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGString.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Random.h>
#include <shogun/mathematics/logdet/SparseMatrixOperator.h>
#include <shogun/mathematics/logdet/ProbingSampler.h>
#include <ColPack/ColPackHeaders.h>

using namespace Eigen;
using namespace ColPack;

namespace shogun
{

CProbingSampler::CProbingSampler() : CTraceSampler()
{
	init();

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CProbingSampler::CProbingSampler(
	CSparseMatrixOperator<float64_t>* matrix_operator, int64_t power,
	EOrderingVariant ordering, EColoringVariant coloring)
	: CTraceSampler(matrix_operator->get_dimension())
{
	init();

	m_power=power;
	m_matrix_operator=matrix_operator;

	std::string str_ordering;
	switch(ordering)
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
	switch(coloring)
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

	m_ordering=SGString<char>(index_t(str_ordering.size()));
	m_coloring=SGString<char>(index_t(str_coloring.size()));
	memcpy(m_ordering.string, str_ordering.data(), str_ordering.size());
	memcpy(m_coloring.string, str_coloring.data(), str_coloring.size());

	SG_REF(m_matrix_operator);

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

void CProbingSampler::init()
{
	m_matrix_operator=NULL;
	m_power=1;

	SG_ADD(&m_probing_vector, "probing_vector", "the probing vector generated"
		" from coloring", MS_NOT_AVAILABLE);

	SG_ADD(&m_power, "matrix-power", "power of the sparse-matrix for coloring",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_ordering, "ordering-variant", "ordering variant for coloring",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_coloring, "coloring-variant", "coloring variant for coloring",
		MS_NOT_AVAILABLE);

	SG_ADD((CSGObject**)&m_matrix_operator, "matrix-operator",
		"the sparse-matrix linear opeator for probing", MS_NOT_AVAILABLE);
}

CProbingSampler::~CProbingSampler()
{
	SG_UNREF(m_matrix_operator);

	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

SGVector<int32_t> CProbingSampler::get_probing_vector() const
{
	return m_probing_vector;
}

void CProbingSampler::precompute()
{
	// do coloring things here and save the probing vector
	SparsityStructure* sp_str=m_matrix_operator->get_sparsity_structure(m_power);

	GraphColoringInterface* Color
		=new GraphColoringInterface(SRC_MEM_ADOLC, sp_str->m_ptr, sp_str->m_num_rows);

	std::string ordering(m_ordering.string, m_ordering.slen);
	std::string coloring(m_coloring.string, m_coloring.slen);
	Color->Coloring(ordering, coloring);

	std::vector<int32_t> vi_VertexColors;
	Color->GetVertexColors(vi_VertexColors);

	REQUIRE(vi_VertexColors.size()==static_cast<uint32_t>(m_dimension),
		"dimension mismatch, %d vs %d!\n", vi_VertexColors.size(), m_dimension);

	m_probing_vector=SGVector<int32_t>(vi_VertexColors.size());

	for (std::vector<int32_t>::iterator it=vi_VertexColors.begin();
		it!=vi_VertexColors.end(); it++)
	{
		index_t i=static_cast<index_t>(std::distance(vi_VertexColors.begin(), it));
		m_probing_vector[i]=*it;
	}

	Map<VectorXi> probe(m_probing_vector.vector, m_probing_vector.vlen);
	m_num_samples=probe.maxCoeff();

	delete sp_str;
	delete Color;
}

SGVector<float64_t> CProbingSampler::sample(index_t idx) const
{
	SGVector<float64_t> s(m_dimension);
	s.set_const(0.0);
	if (idx>=m_num_samples)
		SG_WARNING("idx should be less than %d\n", m_num_samples)
	else
	{
		for (index_t i=0; i<m_dimension; ++i)
		{
			if (m_probing_vector[i]==idx)
			{
				float64_t x=sg_rand->std_normal_distrib();
				s[i]=(x>0)-(x<0);
			}
		}
	}
	return s;
}

}

#endif // HAVE_EIGEN3
#endif // HAVE_COLPACK
