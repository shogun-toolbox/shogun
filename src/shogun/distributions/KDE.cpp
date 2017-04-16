/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Cassio Greco
*/

#include "KDE.h"
#include <shogun/multiclass/KNN.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/distributions/Gaussian.h>
#include <shogun/features/Features.h>

using namespace shogun;

//Constructor of the class. All predefined values are assinged
CKDE::CKDE(int32_t rows,int32_t columns)
{
    this->num_rows=rows;
    this->num_cols=cols;
    this->points = new SGMatrix(rows,columns,true); //Creates a SGMatrix. Will contain the NN Matrix
    this->log_density=0;
    this->bandwidth=1;
}

float64_t CKDE::compute_kde(CFeatures data)
{
	this->points = this->compute_nn(data,this->num_rows,this->num_cols);
	this->log_density = this->compute_pdf(this->points,this->num_rows,this->num_cols,this->bandwidth);

	return this->log_density;
}

//Method that calls the KNN algorithm and the score method
//Calls the KNN algorithm with the data as a parameter
SGMatrix<float64_t> CKDE::compute_nn(CFeatures data, int32_t rows, int32_t cols)
{
	bool worked;
	KNN nn_calculator = new KNN();
	SGMatrix<float64_t> samples = new SGMatrix(rows,cols,true);

	worked = nn_calculator.train_machine(data);

	if worked
		samples=nn_calculator.nearest_neighbors();
	else
		return SG_ERROR(" It was not possible to train the data. ");

	return samples;
}

//Method that computes the PDF of each row
//Returns an SGVector with the value of each kernel function in each set of points
float64_t CKDE::compute_pdf(SGMatrix samples,int32_t rows, int32_t cols,int32_t bandwidth)
{
	int i=0;
	float64_t density = 0;

	Gaussian pdf = new Gaussian();

	for (i=0;i<rows;i++)
		density = density + pdf.compute_log_pdf((samples.get_row_vector[i])/bandwidth);
	
	density = (1/((rows*cols)*bandwidth))*density;
			
	return density;
}

CKDE::~CKDE()
{
    cleanup();
}
