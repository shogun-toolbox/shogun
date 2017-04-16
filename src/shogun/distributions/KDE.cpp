/*
Created by Cassio Greco <cassiodgreco@gmail.com>
*/

#include "KDE.h"
#include <shogun/multiclass/KNN.cpp>
#include <shogun/lib/SGMatrix.cpp>
#include <shogun/distributions/GaussianKernel.cpp> //Kernel
#include <shogun/features/Features.cpp>
//#include <shogun/features/DenseFeatures.h> Does it need dense features?

using namespace shogun;

//Constructor of the class. All predefined values are assinged
CKDE::CKDE(int32_t rows,int32_t columns,CFeatures* datas,bool origin)
{
    	this->algorithm="knn";
    	this->origin=origin;
    	this->num_rows=rows;
    	this->num_cols=cols;
        //this->bandwidth=1.0;
        this->data=datas;
        this->kernel="Gaussian"; //CKernel* when the user will be able to choose the kernel
        this->points(rows,columns,true); //Creates a SGMatrix. Contains the data
        this->points=this->setData(origin,this->points,this->data);
        this->log_density=0;
}

//Method that takes the SGMatrix points, the data that will be used in the algorithm
//and the boolean value origin.
//If origin is true, it means that the data is loaded from a file.
//Is called by the constructor.
SGMatrix CKDE::set_data(bool origin,SGMatrix points,CFeatures data)
{
	if origin
		points.load();
	else
		points=data;
	return points;
}

//Method that calls the KNN algorithm and the score method
CKDE& CKDE::compute()
{
	//Calls the KNN algorithm with the points in the SGMatrix
	this->points=this->points.nearest_neighbors();
	//Calls the method that calculates the kernel weight function
	this->log_density=this->kernel_weight();
	return this->log_density;
}

//Method that computes the Gaussian Kernel(?) weight function
//Returns an SGVector with the value of each kernel function in each set of points in points
SGVector<float64_t> CKDE::kernel_weight()
{
	//Gaussian PDF calculation . Uses SGVector as parameter, not SGMatrix -> needs to be fixed
	SGVector<float64_t> density(this->num_rows*this->num_cols,true);
	int i=0,j=0;
	for (i=0;i<this->num_rows;i++)
		for (j=0;j<this->num_cols;j++)
			density.set_element(this->points.compute(i,j),i+j);
	return density;
}

/*
//Function that calcuates the samples needed for the Gaussian kernel
SGVector<float64_t> CKDE::sample_values(){
	//sample() is a method from Gaussian. Used to calculate the samples needed for the kernel
	SGVector<float64_t> samples = sample(); 
	return samples;
}
*/

//Function for the user to choose which kernel to use. There are so many in the kernel and 
/*
KDE& KDE::choose_kernel(*Ckernel){

}
*/

CKDE::~CKDE()
{
    cleanup();
}
