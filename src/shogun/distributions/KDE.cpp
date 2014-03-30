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
#include <shogun/features/Features.h>

using namespace shogun;

//Constructor of the class. All predefined values are assinged
CKDE::CKDE(int32_t rows,int32_t cols)
{
    m_num_rows = rows;
    m_num_cols = cols;
    m_points = new SGMatrix<float64_t>(rows,cols,true); //Creates a SGMatrix. Will contain the NN Matrix
    m_log_density = 0;
    m_bandwidth = 1;
}

float64_t CKDE::compute_kde(CFeatures* data)
{
	m_points = compute_nn(data,m_num_rows,m_num_cols);
	m_log_density = compute_pdf(m_points,m_num_rows,m_num_cols,m_bandwidth);

	return m_log_density;
}

//Method that calls the KNN algorithm and the score method
//Calls the KNN algorithm with the data as a parameter
SGMatrix<float64_t> CKDE::compute_nn(CFeatures* data, int32_t rows, int32_t cols)
{
	bool worked;
	CKNN* nn_calculator = new CKNN();
	//Auxiliar matrix
	//SGMatrix<index_t> aux = new SGMatrix<index_t>(rows,cols,true);
	SGMatrix<float64_t> samples = new SGMatrix<float64_t>(rows,cols,true);
	//Worked will return true is the nn_calculator was successful in training the data
	nn_calculator->set_k(3);
	worked = nn_calculator->train(data);

	if (worked)
		{
			aux = nn_calculator->nearest_neighbors();

			/*for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					samples[i][j] = (float)aux[i][j];*/
			return samples;
	}
	else
		{
			SG_SERROR(" It was not possible to train the data. ")
			samples.zero();
			return samples;
		}
}

//Method that computes the PDF of each row
//Returns an SGVector with the value of each kernel function in each set of points
float64_t CKDE::compute_pdf(SGMatrix<float64_t> samples,int32_t rows, int32_t cols,int32_t bandwidth)
{
	CGaussian* pdf = new CGaussian();
	float64_t density = 0;
	for (int i = 0;i < rows;i++)
		density = density + pdf->compute_PDF(samples.get_row_vector(i));
	
	density = (1/((rows*cols)*bandwidth))*density;
			
	return density;
}

CKDE::~CKDE()
{
    
}
