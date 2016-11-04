/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/ShiftInvariantKernel.h>
#include <shogun/distance/CustomDistance.h>

using namespace shogun;

CShiftInvariantKernel::CShiftInvariantKernel() : CKernel(0)
{
	register_params();
}

CShiftInvariantKernel::CShiftInvariantKernel(CFeatures *l, CFeatures *r) : CKernel(l, r, 0)
{
	register_params();
	init(l, r);
}

CShiftInvariantKernel::~CShiftInvariantKernel()
{
	cleanup();
	SG_UNREF(m_distance);
}

bool CShiftInvariantKernel::init(CFeatures* l, CFeatures* r)
{
	REQUIRE(m_distance, "The distance instance cannot be NULL!\n");
	CKernel::init(l,r);
	m_distance->init(l, r);
	return init_normalizer();
}

void CShiftInvariantKernel::precompute_distance()
{
	REQUIRE(m_distance, "The distance instance cannot be NULL!\n");
	REQUIRE(m_distance->init(lhs, rhs), "Could not initialize the distance instance!\n");

	SGMatrix<float32_t> dist_mat=m_distance->get_distance_matrix<float32_t>();
	if (m_precomputed_distance==NULL)
	{
		m_precomputed_distance=new CCustomDistance();
		SG_REF(m_precomputed_distance);
	}

	if (lhs==rhs)
		m_precomputed_distance->set_triangle_distance_matrix_from_full(dist_mat.data(), dist_mat.num_rows, dist_mat.num_cols);
	else
		m_precomputed_distance->set_full_distance_matrix_from_full(dist_mat.data(), dist_mat.num_rows, dist_mat.num_cols);
}

void CShiftInvariantKernel::cleanup()
{
	SG_UNREF(m_precomputed_distance);
	m_precomputed_distance=NULL;
	CKernel::cleanup();
	m_distance->cleanup();
}

EDistanceType CShiftInvariantKernel::get_distance_type() const
{
	REQUIRE(m_distance, "The distance instance cannot be NULL!\n");
	return m_distance->get_distance_type();
}

float64_t CShiftInvariantKernel::distance(int32_t a, int32_t b) const
{
	REQUIRE(m_distance, "The distance instance cannot be NULL!\n");
	if (m_precomputed_distance!=NULL)
		return m_precomputed_distance->distance(a, b);
	else
		return m_distance->distance(a, b);
}

void CShiftInvariantKernel::register_params()
{
	SG_ADD((CSGObject**) &m_distance, "m_distance", "Distance to be used.", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &m_precomputed_distance, "m_precomputed_distance", "Precomputed istance to be used.", MS_NOT_AVAILABLE);

	m_distance=NULL;
	m_precomputed_distance=NULL;
}

void CShiftInvariantKernel::set_precomputed_distance(CCustomDistance* precomputed_distance)
{
	REQUIRE(precomputed_distance, "The precomputed distance instance cannot be NULL!\n");
	SG_REF(precomputed_distance);
	SG_UNREF(m_precomputed_distance);
	m_precomputed_distance=precomputed_distance;
}

CCustomDistance* CShiftInvariantKernel::get_precomputed_distance() const
{
	REQUIRE(m_precomputed_distance, "The precomputed distance instance cannot be NULL!\n");
	SG_REF(m_precomputed_distance);
	return m_precomputed_distance;
}
