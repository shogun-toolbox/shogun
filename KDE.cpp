/*
Created by Cassio Greco <cassiodgreco@gmail.com>
*/

#include "KDE.h"
#include <shogun/multiclass/KNN.cpp>
#include <shogun/lib/SGMatrix.cpp> 
#include <shogun/distributions/Gaussian.cpp> //Kernel

using namespace shogun;

//Constructor of the class. All predefined values are assinged
KDE::KDE(rows,columns){ 
    	this->algorithm = "knn";
        //this->bandwidth = 1.0;
        this->kernel = "Gaussian"; //CKernel* when the user will be able to choose the kernel
        //this->metric = metric;
        this->points = this->points.SGMatrix(rows,columns,true); //Creates a SGMatrix. Contains the data
        this->log_density = 0;

}


//Method that calls the KNN algorithm and the score method
KDE& KDE::compute(){
	//Calls the KNN algorithm with the points in the SGMatrix
	this->points = this->nearest_neighbors(); 
	this->log_density = this->score();
	return *this;
}

//Method that computes the Gaussian log_PDF
float64_t KDE::score(){
	//Gaussian PDF calculation . Uses SGVector as parameter, not SGMatrix -> needs to be fixed
	float64_t density = compute_log_PDF(this->points); 
	return density;
}

//Function that calcuates the samples needed for the Gaussian kernel
SGVector<float64_t> KDE::sample_values(){
	//sample() is a method from Gaussian. Used to calculate the samples needed for the kernel
	SGVector<float64_t> samples = sample(); 
	return samples;
}

//Function for the user to choose which kernel to use. There are so many in the kernel and 
/*
KDE& KDE::choose_kernel(*Ckernel){

}
*/

KDE::~KDE()
{
    cleanup();
}
