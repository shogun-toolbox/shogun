/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2014 Cassio Greco
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
*/

#include "KDE.h"
#include <shogun/multiclass/KNN.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/io/SGIO.h>
#include <shogun/distributions/Gaussian.h>
#include <shogun/distributions/Distribution.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

//Constructor of the class. All predefined values are assinged
CKDE::CKDE(index_t num_points)
{
	m_values = SGVector<float64_t>(num_points,true);
    m_log_density = 0;
    m_bandwidth = 1;
    m_pdf = new CGaussian();
    m_nn_calculator = new CKNN();
}

CKDE::~CKDE()
{
    
}

float64_t CKDE::compute_kde(CFeatures* data,int32_t k)
{
	m_values = compute_nn(data,k);
	m_log_density = compute_pdf();

	return m_log_density;
}

//Method that calls the KNN algorithm and the score method
//Calls the KNN algorithm with the data as a parameter
SGVector<float64_t> CKDE::compute_nn(CFeatures* data,int32_t k)
{
	CLabels *labels;
	//Worked will return true is the nn_calculator was successful in training the data
	m_nn_calculator->set_k(k);

	labels = m_nn_calculator->apply(data);

	SG_UNREF(m_nn_calculator);
	return labels->get_values();
}

//Method that computes the PDF of each row
//Returns an SGVector with the value of each kernel function in each set of points
float64_t CKDE::compute_pdf()
{
  	float64_t density = 0;

  	density = m_pdf->compute_PDF(m_values.get())/m_bandwidth;
  	density *= (1/(m_values.size()*m_bandwidth));

	SG_UNREF(m_pdf);
	return density;
}
