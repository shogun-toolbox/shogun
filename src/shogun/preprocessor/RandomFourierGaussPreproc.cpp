/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010-2011 Alexander Binder
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010-2011 Berlin Institute of Technology
 */

#include <shogun/preprocessor/RandomFourierGaussPreproc.h>
#include <cmath>

using namespace shogun;

void CRandomFourierGaussPreproc::copy(const CRandomFourierGaussPreproc & feats) {

	dim_input_space = feats.dim_input_space;
	dim_feature_space = feats.dim_feature_space;
	kernelwidth=feats.kernelwidth;


	if(feats.randomcoeff_additive==NULL)
	{
		SG_FREE(randomcoeff_additive);
		randomcoeff_additive=NULL;
	}
	else
	{
		randomcoeff_additive = SG_MALLOC(float64_t, dim_feature_space);
		std::copy(feats.randomcoeff_additive,feats.randomcoeff_additive+dim_feature_space,randomcoeff_additive);
	}

	if(feats.randomcoeff_multiplicative==NULL)
	{
		SG_FREE(randomcoeff_multiplicative);
		randomcoeff_multiplicative=NULL;
	}
	else
	{
		randomcoeff_multiplicative=SG_MALLOC(float64_t, dim_feature_space*dim_input_space);
		std::copy(feats.randomcoeff_multiplicative,feats.randomcoeff_multiplicative+dim_feature_space*dim_input_space,randomcoeff_multiplicative);
	}


}

CRandomFourierGaussPreproc::CRandomFourierGaussPreproc() :
	CPreprocessor() {
	dim_feature_space = 0;
	dim_input_space = 0;
	kernelwidth=0;

	randomcoeff_multiplicative=NULL;
	randomcoeff_additive=NULL;




	//m_parameter is inherited from CSGObject,
	//serialization initialization
	if(m_parameters)
	{
		SG_ADD(&dim_input_space, "dim_input_space",
		    "Dimensionality of the input space.", MS_NOT_AVAILABLE);
		SG_ADD(&dim_feature_space, "dim_feature_space",
		    "Dimensionality of the feature space.", MS_NOT_AVAILABLE);

		SG_ADD(&kernelwidth, "kernelwidth", "Kernel width.", MS_AVAILABLE);

		m_parameters->add_vector(&randomcoeff_additive,&dim_feature_space,"randomcoeff_additive");
		m_parameters->add_matrix(&randomcoeff_multiplicative,&dim_feature_space,&dim_input_space,"randomcoeff_multiplicative");
	}

}

CRandomFourierGaussPreproc::CRandomFourierGaussPreproc(
		const CRandomFourierGaussPreproc & feats) :
	CPreprocessor () {

	randomcoeff_multiplicative=NULL;
	randomcoeff_additive=NULL;

	//m_parameter is inherited from CSGObject,
	//serialization initialization
	if(m_parameters)
	{
		SG_ADD(&dim_input_space, "dim_input_space",
		    "Dimensionality of the input space.", MS_NOT_AVAILABLE);
		SG_ADD(&dim_feature_space, "dim_feature_space",
		    "Dimensionality of the feature space.", MS_NOT_AVAILABLE);

		SG_ADD(&kernelwidth, "kernelwidth", "Kernel width.", MS_AVAILABLE);
		SG_ADD(&kernelwidth, "cur_kernelwidth", "Kernel width.", MS_AVAILABLE);

		m_parameters->add_vector(&randomcoeff_additive,&dim_feature_space,"randomcoeff_additive");
		m_parameters->add_matrix(&randomcoeff_multiplicative,&dim_feature_space,&dim_input_space,"randomcoeff_multiplicative");
	}

	copy(feats);
}

CRandomFourierGaussPreproc::~CRandomFourierGaussPreproc() {

	SG_FREE(randomcoeff_multiplicative);
	SG_FREE(randomcoeff_additive);

}

EFeatureClass CRandomFourierGaussPreproc::get_feature_class() {
	return C_DENSE;
}

EFeatureType CRandomFourierGaussPreproc::get_feature_type() {
	return F_DREAL;
}

int32_t CRandomFourierGaussPreproc::get_dim_feature_space() const {
	return ((int32_t) dim_feature_space);
}


int32_t CRandomFourierGaussPreproc::get_dim_input_space() const {
	return ((int32_t) dim_input_space);
}


float64_t CRandomFourierGaussPreproc::get_kernelwidth( ) const {
	return (kernelwidth);
}



bool CRandomFourierGaussPreproc::test_rfinited() const {

	if (dim_input_space<=0)
	{
		return false;
	} 
	if (dim_feature_space<=0)
	{
		return false;
	} 
	if (kernelwidth<=0)
	{
		return false;
	} 

	if(!randomcoeff_additive)
	{
		return false;
	} 

	if(!randomcoeff_multiplicative)
	{
		return false;
	} 

	return true;

}

bool CRandomFourierGaussPreproc::init_randomcoefficients_from_scratch() {
	if (dim_feature_space <= 0) {
		SG_ERROR(
				"bool CRandomFourierGaussPreproc::init_randomcoefficients_from_scratch(): dim_feature_space<=0 is not allowed\n");
	}
	if (dim_input_space <= 0) {
		SG_ERROR(
				"bool CRandomFourierGaussPreproc::init_randomcoefficients_from_scratch(): dim_input_space<=0 is not allowed\n");
	}
	if (kernelwidth <= 0) {
		SG_ERROR(
				"bool CRandomFourierGaussPreproc::init_randomcoefficients_from_scratch(): kernelwidth<=0 is not allowed\n");
	}


	SG_INFO("initializing randomcoefficients \n") 

	float64_t pi = 3.14159265;


	SG_FREE(randomcoeff_multiplicative);
	randomcoeff_multiplicative=NULL;
	SG_FREE(randomcoeff_additive);
	randomcoeff_additive=NULL;

	randomcoeff_additive=SG_MALLOC(float64_t, dim_feature_space);
	randomcoeff_multiplicative=SG_MALLOC(float64_t, dim_feature_space*dim_input_space);



	for (int32_t  i = 0; i < dim_feature_space; ++i) {
		randomcoeff_additive[i] = CMath::random((float64_t) 0.0, 2 * pi);
	}

	for (int32_t  i = 0; i < dim_feature_space; ++i) {
		for (int32_t k = 0; k < dim_input_space; ++k) {
			float64_t x1,x2;
			float64_t s = 2;
			while ((s >= 1) ) {
				// Marsaglia polar for gaussian
				x1 = CMath::random((float64_t) -1.0, (float64_t) 1.0);
				x2 = CMath::random((float64_t) -1.0, (float64_t) 1.0);
				s=x1*x1+x2*x2;
			}

			// =  x1/CMath::sqrt(val)* CMath::sqrt(-2*CMath::log(val));
			randomcoeff_multiplicative[i*dim_input_space+k] =  x1*CMath::sqrt(-2*CMath::log(s)/s )/kernelwidth;
		}
	}

	SG_INFO("finished: initializing randomcoefficients \n") 

	return true;
}

void CRandomFourierGaussPreproc::get_randomcoefficients(
		float64_t ** randomcoeff_additive2,
		float64_t ** randomcoeff_multiplicative2, int32_t *dim_feature_space2,
		int32_t *dim_input_space2, float64_t* kernelwidth2) const {

	ASSERT(randomcoeff_additive2)
	ASSERT(randomcoeff_multiplicative2)

	if (!test_rfinited()) {
		*dim_feature_space2 = 0;
		*dim_input_space2 = 0;
		*kernelwidth2=1;
		*randomcoeff_additive2 = NULL;
		*randomcoeff_multiplicative2 = NULL;
		return;
	}

	*dim_feature_space2 = dim_feature_space;
	*dim_input_space2 = dim_input_space;
	*kernelwidth2=kernelwidth;

	*randomcoeff_additive2 = SG_MALLOC(float64_t, dim_feature_space);
	*randomcoeff_multiplicative2 = SG_MALLOC(float64_t, dim_feature_space*dim_input_space);

	std::copy(randomcoeff_additive, randomcoeff_additive+dim_feature_space,
			*randomcoeff_additive2);
	std::copy(randomcoeff_multiplicative, randomcoeff_multiplicative+dim_feature_space*dim_input_space,
			*randomcoeff_multiplicative2);


}

void CRandomFourierGaussPreproc::set_randomcoefficients(
		float64_t *randomcoeff_additive2,
		float64_t * randomcoeff_multiplicative2,
		const int32_t dim_feature_space2, const int32_t dim_input_space2, const float64_t kernelwidth2) {
	dim_feature_space = dim_feature_space2;
	dim_input_space = dim_input_space2;
	kernelwidth=kernelwidth2;

	SG_FREE(randomcoeff_multiplicative);
	randomcoeff_multiplicative=NULL;
	SG_FREE(randomcoeff_additive);
	randomcoeff_additive=NULL;


	if( (dim_feature_space>0) && (dim_input_space>0) )
	{
	randomcoeff_additive=SG_MALLOC(float64_t, dim_feature_space);
	randomcoeff_multiplicative=SG_MALLOC(float64_t, dim_feature_space*dim_input_space);

	std::copy(randomcoeff_additive2, randomcoeff_additive2
			+ dim_feature_space, randomcoeff_additive);
	std::copy(randomcoeff_multiplicative2, randomcoeff_multiplicative2
			+ dim_feature_space*dim_input_space, randomcoeff_multiplicative);
	}

}

bool CRandomFourierGaussPreproc::set_parameters(const int32_t dim_input_space2, const int32_t dim_feature_space2, const float64_t kernelwidth2) {

	dim_input_space=dim_input_space2;
	dim_feature_space=dim_feature_space2;
	kernelwidth=kernelwidth2;


	return true;
}

/*
SGVector<float64_t> CRandomFourierGaussPreproc::apply_to_feature_vector(SGVector<float64_t> vector)
{
	if (!test_rfinited()) {
		SG_ERROR(
				"float64_t * CRandomFourierGaussPreproc::apply_to_feature_vector(...): test_rfinited()==false: you need to call before CRandomFourierGaussPreproc::init (...)  \n");
	}

	int32_t num_features =	vector.size();
	if (num_features!=dim_input_space)
	{
		SG_ERROR(
						"float64_t * CRandomFourierGaussPreproc::apply_to_feature_vector(...): num_features!=dim_input_space is not allowed\n");
	}


	float64_t val = CMath::sqrt(2.0 / dim_feature_space);
	float64_t *res = SG_MALLOC(float64_t, dim_feature_space);

	for (int32_t od = 0; od < dim_feature_space; ++od) {
		res[od] = val * cos(randomcoeff_additive[od] + SGVector<float64_t>::dot(vector.vector,
				randomcoeff_multiplicative+od*dim_input_space, dim_input_space));
	}

	return SGVector<float64_t>(res,dim_feature_space);
}
*/


bool CRandomFourierGaussPreproc::check_applicability_to_feature(CFeatures* features)
{
	if(!(
		(features->get_feature_class()==C_SPARSE)
		||(features->get_feature_class()==C_DENSE)
		||(features->get_feature_class()==C_BINNED_DOT)
		||(features->get_feature_class()==C_COMBINED_DOT)
	))
	{
		SG_ERROR("CRandomFourierGaussPreproc::check_applicability_to_feature(...):features->get_feature_class()!=C_SPARSE, C_DENSE, C_COMBINED_DOT or C_BINNING.Don't know how to apply a cosine on other features!\n");
		return false;
	}
	 
	if(!(
  		(features->get_feature_type()==F_SHORTREAL)
		||(features->get_feature_type()==F_DREAL)
		||(features->get_feature_type()==F_LONGREAL)
		||(features->get_feature_type()==F_INT)
		||(features->get_feature_type()==F_UINT)
		||(features->get_feature_type()==F_LONG)
		||(features->get_feature_type()==F_ULONG)
	))
	{
		SG_ERROR("CRandomFourierGaussPreproc::check_applicability_to_feature(...):features->get_feature_type() is not real or integer! It is likely unusual to apply a cosine on these features\n");
		return false;
	}

	return true;
}

CDenseFeatures<float64_t>* CRandomFourierGaussPreproc::apply_to_dotfeatures_sparse_or_dense_with_real(CDotFeatures* features)
{

	if(false==check_applicability_to_feature( features))
	{
		SG_ERROR("CRandomFourierGaussPreproc::check_applicability_to_feature(...):feature_class must be of type C_SPARSE, C_DENSE or C_BINNING, feature_type must be integer or real\n");
	}



	int32_t num_features = features->get_dim_feature_space();
	if (num_features!=dim_input_space)
	{
		SG_ERROR("class CRandomFourierGaussPreproc: num_features!=dim_input_space is not allowed\n");
	}

	int32_t num_samples=features->get_num_vectors();
	SGMatrix<float64_t> outmatrix(dim_feature_space,num_samples);

	float64_t val = CMath::sqrt(2.0 / dim_feature_space);
	for (int32_t vecind = 0; vecind < num_samples; vecind++)
	{
		for (int32_t od = 0; od < dim_feature_space; ++od)
		{
			float64_t tmpval=randomcoeff_additive[od]+features->dense_dot (vecind, randomcoeff_multiplicative+od*dim_input_space, dim_input_space);
			outmatrix.matrix[od+dim_feature_space*vecind]=val*cos(tmpval);
		}
	}

	CDenseFeatures<float64_t> * result = new CDenseFeatures<float64_t>(outmatrix);

	return(result);

}

bool CRandomFourierGaussPreproc::init(CFeatures *features)
{
return true;
}


void CRandomFourierGaussPreproc::cleanup()
{

}
