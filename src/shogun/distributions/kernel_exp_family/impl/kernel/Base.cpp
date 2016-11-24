/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Heiko Strathmann
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>

#include "Base.h"

using namespace shogun;
using namespace shogun::kernel_exp_family_impl::kernel;
using namespace Eigen;

index_t Base::get_num_dimensions() const
{
	return m_lhs.num_rows;
}

index_t Base::get_num_lhs() const
{
	return m_lhs.num_cols;
}

void Base::set_rhs(SGMatrix<float64_t> rhs)
{
	m_rhs = rhs;
}

void Base::set_rhs(SGVector<float64_t> rhs)
{
	set_rhs(SGMatrix<float64_t>(rhs));
}

void Base::set_lhs(SGMatrix<float64_t> lhs)
{
	m_lhs = lhs;
}

void Base::set_lhs(SGVector<float64_t> lhs)
{
	set_lhs(SGMatrix<float64_t>(lhs));
}

SGMatrix<float64_t> Base::get_lhs()
{
	return m_lhs;
}

void Base::get_lhs(index_t idx, SGVector<float64_t>& dst)
{
	auto D = get_num_dimensions();
	memcpy(dst.vector, m_lhs.get_column_vector(idx), D*sizeof(float64_t));
}

void Base::get_rhs(index_t idx, SGVector<float64_t>& dst)
{
	auto D = get_num_dimensions();
	memcpy(dst.vector, m_rhs.get_column_vector(idx), D*sizeof(float64_t));
}

index_t Base::get_num_rhs() const
{
	return m_rhs.num_cols;
}

Base::Base()
{
}
