/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Cassio Greco
*/

#ifndef KDE_H
#define KDE_H

#include <shogun/multiclass/KNN.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/Features.h>
#include <shogun/distributions/Gaussian.h>

using namespace shogun;

class CKDE : public SGObject
{
    public:

        //Constructor
        CKDE(int32_t,int32_t);

        /* Method takes CFeatures (data) as an argument.
         * 
         * Calls the methods compute_nn and compute_pdf
         * that will respectively calculate the NN matrix
         * using the data and the PDF of the NN matrix
        */
        float64_t compute_kde(CFeatures);

        /* Method takes CFeatures (data) and the ammount
         * of rows and columns of the points SGMatrix.
         * 
         * Creates the boolean worked that will be true
         * if the nn_calculator was able to train the data.
         * If worked is true, the nearest neighbor algorithm
         * is executed and saved in the samples SGMatrix,
         * which will be returned.
        */
        SGMatrix<float64_t> compute_nn(CFeatures,int32_t,int32_t);

        /* Method takes in SGMatrix samples from compute_nn,
         * the number of rows, columns and bandwidth as argument.
         *
         * Density is iteratively calculated for every row in the
         * SGMatrix samples. To calculate density, it will be the
         * sum of itself and the log pdf of each row, divided by the
         * bandwitdth, calculated by the Gaussian method. The final 
         * value of density is then multiplied by 1/(n*bandwidth).
         * Density is then returned.
        */
        float64_t compute_pdf(SGMatrix<float64_t>,int32_t,int32_t,int32_t);

        //Destructor
        virtual ~CKDE();

    protected:

    	SGMatrix points;

    	float64_t log_density;

        int32_t num_rows,num_cols,bandwidth;

    private: 
};

#endif // KDE_H
