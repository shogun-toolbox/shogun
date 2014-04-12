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
#include <shogun/labels/Labels.h>
#include <shogun/distributions/Distribution.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;


/** This class is used for the calculation of the Kernel Density Estimation of a given 
 * data.
 * First it uses the K-Nearest Neighbor algorithm to classify the data into the correct
 * bin. Secondly the Probability Density Function is then calculated. Thirdly, the function: 
 * \f$\frac{1}{(n \times h\)} \times \sum_{K(frac{x-xi}{h}}^)\f$ is calculated, where:
 * h = bandwidth, K equals the Gaussian PDF calculation (the kernel used), and n = The number of features.
*/


class CKDE : public CDistribution
{
    public:

        /** @brief sets the values of the private variables.
         * @param num_points
         */
        CKDE(index_t num_points);

        /** Destructor */
        virtual ~CKDE();

        /** @brief Calls the methods compute_nn and compute_pdf
         *that will respectively calculate the NN matrix
         *using the data and the PDF of the NN matrix.
         *
         * @param takes CFeatures (data) as an argument and
         *k, which is the k parameter in the Nearest Neighbor algorithm.
         *
         * @return the value of the Kernel Density Estimation function
         *of the given data.
         */
        float64_t compute_kde(CFeatures* data,int32_t k);

        /** @brief The nearest neighbor algorithm is executed and saved
         * in the values SGVector.
         *
         * @param Method takes CFeatures (data)
         * k, which is the k parameter in the Nearest Neighbor algorithm.
         *
         * @return the SGVector values.
         */
        SGVector<float64_t> compute_nn(CFeatures* data,int32_t k);

        /** @brief Density is calculated based on the SGVector m_values.
         * To calculate density, it will be the log pdf of a vector, 
         * divided by the bandwidth, calculated by the Gaussian method.
         * The final value of density is then multiplied by 1/(n*bandwidth).
         *            
         * @return density.
         */
        float64_t compute_pdf();

        /** @brief The method sets the value of m_bandwidth equal to the
         * value of the parameter bandwidth.
         *
         * @param takes an integer as an argument.        
         */
        void set_bandwidth(int32_t bandwidth) {m_bandwidth = bandwidth;}

        /* @return the value of m_bandwidth.
         */
        int32_t get_bandwidth() {return m_bandwidth;}

        /** learn distribution
         *
         * @param data training data (parameter can be avoided if distance or
         * kernel-based classifiers are used and distance/kernels are
         * initialized with train data)
         *
         * @return whether training was successful
         */
        virtual bool train(CFeatures* data=NULL)=0;

        /** get number of parameters in model
         *
         * abstract base method
         *
         * @return number of parameters in model
         */
        virtual int32_t get_num_model_parameters()=0;

        /** get model parameter (logarithmic)
         *
         * abstract base method
         *
         * @return model parameter (logarithmic)
         */
        virtual float64_t get_log_model_parameter(int32_t num_param)=0;

        /** get partial derivative of likelihood function (logarithmic)
         *
         * abstract base method
         *
         * @param num_param derivative against which param
         * @param num_example which example
         * @return derivative of likelihood (logarithmic)
         */
        virtual float64_t get_log_derivative(
            int32_t num_param, int32_t num_example)=0;

        /** compute log likelihood for example
         *
         * abstract base method
         *
         * @param num_example which example
         * @return log likelihood for example
         */
        virtual float64_t get_log_likelihood_example(int32_t num_example)=0;

    private: 

        /* SGVector that will contain the data after it has been processed by the Nearest Neighbor algorithm */
        SGVector<float64_t> m_values;

        /* Variable that will contain the value of the Kernel Density Estimation */
        float64_t m_log_density;

        /* The bandwidth is the width of the kernel used (similar to a histogram's bin)*/
        int32_t m_bandwidth;

        /* Instance of CKNN that will train the data and calculate the nearest neighbor algorithm */
        CKNN* m_nn_calculator;

        /* Instance of CGaussian that will calculate the Probability Density Function */
        CGaussian* m_pdf;
};

#endif // KDE_H