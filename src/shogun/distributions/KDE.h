/*
Created by Cassio Greco <cassiodgreco@gmail.com>
*/

#ifndef KDE_H
#define KDE_H

#include <shogun/multiclass/KNN.h>
#include <shogun/lib/SGMatrix.cpp>

using namespace shogun;



class KDE: public CKNN{
    public:
        KDE(*); // Constructor
        virtual ~KDE(); // Destructor
        KDE compute();
        float64_t score();
        SGVector<float64_t> sample_values();
    protected:
    	//float bandwidth;
    	char *algorithm;
    	char *kernel
    	//char metric;
    	SGMatrix points; //Do I specify the size?
    	float64_t log_density;
    private: 
};

#endif // KDE_H