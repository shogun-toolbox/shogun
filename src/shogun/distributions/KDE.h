/*
Created by Cassio Greco <cassiodgreco@gmail.com>
*/

#ifndef KDE_H
#define KDE_H

#include <shogun/multiclass/KNN.h>
#include <shogun/lib/SGMatrix.cpp>
#include <shogun/features/Features.cpp>

using namespace shogun;

class CKDE{
    public:
        CKDE(int32_t,int32_t,CFeatures*,bool);//Constructor
        virtual ~CKDE();//Destructor
        CKDE compute();
        SGMatrix set_data(bool, SGMatrix, CFeatures);
        SGVector<float64_t> kernel_weight();
        //SGVector<float64_t> sample_values();
    protected:
    	//float bandwidth;
    	char *algorithm;
    	char *kernel
    	SGMatrix points; //Do I specify the size?
    	SGVector<float64_t> log_density;
        CFeatures data;
        bool origin; //True if matrix comes from a file. False if it is not.
        int32_t num_rows,num_cols;
    private: 
};

#endif // KDE_H
