/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Alexander Binder
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/classifier/mkl/MKLMultiClassOptimizationBase.h>

using namespace shogun;

MKLMultiClassOptimizationBase::MKLMultiClassOptimizationBase()
{

}
MKLMultiClassOptimizationBase::~MKLMultiClassOptimizationBase()
{

}



void MKLMultiClassOptimizationBase::setup(const int32_t numkernels2)
{
	SG_ERROR("class MKLMultiOptimizationBase, method not implemented in derivedclass");

}

void MKLMultiClassOptimizationBase::set_mkl_norm(float64_t norm)
{
	//deliberately no error here
	SG_WARNING("class MKLMultiOptimizationBase, method set_mkl_norm() not implemented in derived class, has no effect");
}

void MKLMultiClassOptimizationBase::addconstraint(const ::std::vector<float64_t> & normw2,
		const float64_t sumofpositivealphas)
{
	SG_ERROR("class MKLMultiOptimizationBase, method not implemented in derivedclass");

}



void MKLMultiClassOptimizationBase::computeweights(std::vector<float64_t> & weights2)
{
	SG_ERROR("class MKLMultiOptimizationBase, method not implemented in derivedclass");
}
