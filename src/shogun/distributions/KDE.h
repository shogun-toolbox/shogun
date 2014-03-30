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

#include <shogun/lib/config.h>

#ifndef KDE_H
#define KDE_H

#include <shogun/multiclass/KNN.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/Features.h>
#include <shogun/distributions/Gaussian.h>

using namespace shogun;

/* This class is used for the calculation of the Kernel Density Estimation of a given 
 * data.
 * First it uses the K-Nearest Neighbor algorithm to classify the data into the correct
 * bin. Secondly the Probability Density Function is then calculated over every row of
 * the data SGMatrix. Thirdly, the function: (1/(n*h)) * SUM of Kernel(x-xi/h), where:
 * h = bandwidth, Kernel equals the Gaussian PDF calculation, and n = The number of features.
 *
 * CKDE inherits from CKNN to have access to the protected method nearest_neighbors()
*/


class CKDE
{
    public:

        /*Constructor
         * @param rows
         * @param cols
         */
        CKDE(int32_t rows,int32_t cols);

        /* @param  Method takes CFeatures (data) as an argument.
         * 
         *         Calls the methods compute_nn and compute_pdf
         *         that will respectively calculate the NN matrix
         *         using the data and the PDF of the NN matrix.
        */
        float64_t compute_kde(CFeatures* data);

        /* @param Method takes CFeatures (data) and the ammount
         *        of rows and columns of the points SGMatrix.
         * 
         *        If worked is true, the nearest neighbor algorithm
         *        is executed and saved in the samples SGMatrix.
         *        
         *
         * @result Returns the SGMatrix samples
        */
        SGMatrix<float64_t> compute_nn(CFeatures* data,int32_t rows,int32_t cols);

        /* @param Method takes in SGMatrix samples from compute_nn,
         *        the number of rows, columns and bandwidth as argument.
         *
         *        Density is iteratively calculated for every row in the
         *        SGMatrix samples. To calculate density, it will be the
         *        sum of itself and the log pdf of each row, divided by the
         *        bandwidth, calculated by the Gaussian method. The final 
         *        value of density is then multiplied by 1/(n*bandwidth).
         *        
         * @result density is returned.
        */
        float64_t compute_pdf(SGMatrix<float64_t> samples,int32_t rows,int32_t cols,int32_t bandwidth);

        //Destructor
        virtual ~CKDE();

    private: 

        /* SGMatrix that will contain the data after it has been processed by the Nearest Neighbor algorithm */
        SGMatrix<float64_t> m_points;

        /* Variable that will contain the value of the Kernel Density Estimation */
        float64_t m_log_density;

        /* Number of rows of the m_points SGMatrix */
        int32_t m_num_rows;

        /* Number of columns of the m_points SGMatrix */
        int32_t m_num_cols;

        /* Variable that contains the bandwidth of the PDF function */
        int32_t m_bandwidth;
};

#endif // KDE_H
