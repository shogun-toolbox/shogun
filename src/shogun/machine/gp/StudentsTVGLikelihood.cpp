/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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
 *
 * Code adapted from 
 * http://hannes.nickisch.org/code/approxXX.tar.gz
 * and the reference paper is
 * Nickisch, Hannes, and Carl Edward Rasmussen.
 * "Approximations for Binary Gaussian Process Classification."
 * Journal of Machine Learning Research 9.10 (2008).
 *
 * This code specifically adapted from function in approxKL.m
 */

#include <shogun/machine/gp/StudentsTVGLikelihood.h>

#ifdef HAVE_EIGEN3
#include <shogun/machine/gp/StudentsTLikelihood.h>

using namespace Eigen;

namespace shogun
{

CStudentsTVGLikelihood::CStudentsTVGLikelihood()
	: CNumericalVGLikelihood()
{
	m_sigma = 1.0;
	m_df = 3.0;
	init();
}

CStudentsTVGLikelihood::CStudentsTVGLikelihood(float64_t sigma, float64_t df)
	: CNumericalVGLikelihood()
{
	REQUIRE(sigma>0.0, "Scale parameter must be greater than zero\n")
	REQUIRE(df>1.0, "Number of degrees of freedom must be greater than one\n")

	m_sigma=sigma;
	m_df=df;
	init();
}

CStudentsTVGLikelihood::~CStudentsTVGLikelihood()
{
}

void CStudentsTVGLikelihood::init_likelihood()
{
	set_likelihood(new CStudentsTLikelihood(m_sigma, m_df));
}

void CStudentsTVGLikelihood::init()
{
	init_likelihood();
	SG_ADD(&m_df, "df", "Degrees of freedom", MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD(&m_sigma, "sigma", "Scale parameter", MS_AVAILABLE, GRADIENT_AVAILABLE);
}

} /* namespace shogun */

#endif /* HAVE_EIGEN3 */
