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

ShiftInvariantKernel::ShiftInvariantKernel() : Kernel(0)
{
	register_params();
}

ShiftInvariantKernel::ShiftInvariantKernel(const std::shared_ptr<Features >&l, const std::shared_ptr<Features >&r) : Kernel(l, r, 0)
{
	register_params();
	init(l, r);
}

ShiftInvariantKernel::~ShiftInvariantKernel()
{
	cleanup();
	
}

bool ShiftInvariantKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	require(m_distance, "The distance instance cannot be NULL!");
	Kernel::init(l,r);
	m_distance->init(l, r);
	return init_normalizer();
}

void ShiftInvariantKernel::precompute_distance()
{
	require(m_distance, "The distance instance cannot be NULL!");
	require(m_distance->init(lhs, rhs), "Could not initialize the distance instance!");

	SGMatrix<float32_t> dist_mat=m_distance->get_distance_matrix<float32_t>();
	if (m_precomputed_distance==NULL)
	{
		m_precomputed_distance=std::make_shared<CustomDistance>();
		
	}

	if (lhs==rhs)
		m_precomputed_distance->set_triangle_distance_matrix_from_full(dist_mat.data(), dist_mat.num_rows, dist_mat.num_cols);
	else
		m_precomputed_distance->set_full_distance_matrix_from_full(dist_mat.data(), dist_mat.num_rows, dist_mat.num_cols);
}

void ShiftInvariantKernel::cleanup()
{
	
	m_precomputed_distance=NULL;
	Kernel::cleanup();
	m_distance->cleanup();
}

EDistanceType ShiftInvariantKernel::get_distance_type() const
{
	require(m_distance, "The distance instance cannot be NULL!");
	return m_distance->get_distance_type();
}

float64_t ShiftInvariantKernel::distance(int32_t a, int32_t b) const
{
	require(m_distance, "The distance instance cannot be NULL!");
	if (m_precomputed_distance!=NULL)
		return m_precomputed_distance->distance(a, b);
	else
		return m_distance->distance(a, b);
}

void ShiftInvariantKernel::register_params()
{
	SG_ADD((std::shared_ptr<SGObject>*) &m_distance, "m_distance", "Distance to be used.");
	SG_ADD((std::shared_ptr<SGObject>*) &m_precomputed_distance, "m_precomputed_distance", "Precomputed istance to be used.");

	m_distance=NULL;
	m_precomputed_distance=NULL;
}

void ShiftInvariantKernel::set_precomputed_distance(const std::shared_ptr<CustomDistance>& precomputed_distance)
{
	require(precomputed_distance, "The precomputed distance instance cannot be NULL!");
	m_precomputed_distance=precomputed_distance;
}

std::shared_ptr<CustomDistance> ShiftInvariantKernel::get_precomputed_distance() const
{
	require(m_precomputed_distance, "The precomputed distance instance cannot be NULL!");
	return m_precomputed_distance;
}
