/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Alexander Binder
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 *
 * Update to patch 0.10.0 - thanks to Eric aka Yoo (thereisnoknife@gmail.com)
 *
 */

#include <vector>
#include <shogun/classifier/mkl/MKLMulticlassGLPK.h>
#ifdef USE_GLPK
#include <glpk.h>
#endif


using namespace shogun;

MKLMulticlassGLPK::MKLMulticlassGLPK()
{
	numkernels = 0;
#ifdef USE_GLPK
	//makes glpk quiet
	glp_term_out(GLP_OFF);
	linearproblem=NULL;
#endif
}
MKLMulticlassGLPK::~MKLMulticlassGLPK()
{
#if defined(USE_GLPK)
	if (linearproblem)
	{
      glp_delete_prob((glp_prob*) linearproblem);
      linearproblem=NULL;
	}

#endif
}

MKLMulticlassGLPK MKLMulticlassGLPK::operator=(MKLMulticlassGLPK & gl)
{
	SG_ERROR(
         " MKLMulticlassGLPK MKLMulticlassGLPK::operator=(...): must "
			"not be called, glpk structure is currently not copyable");
	return (*this);

}
MKLMulticlassGLPK::MKLMulticlassGLPK(MKLMulticlassGLPK & gl)
{
	SG_ERROR(
         " MKLMulticlassGLPK::MKLMulticlassGLPK(MKLMulticlassGLPK & gl):"
			" must not be called, glpk structure is currently not copyable");

}

void MKLMulticlassGLPK::setup(const int32_t numkernels2)
{
#if defined(USE_GLPK)
	numkernels=numkernels2;
	if (numkernels<=1)
	{
		SG_ERROR("void glpkwrapper::setup(const int32_tnumkernels): input "
				"numkernels out of bounds: %d\n",numkernels);
	}

	if (!linearproblem)
	{
		linearproblem=glp_create_prob();
	}

   glp_set_obj_dir((glp_prob*)linearproblem, GLP_MAX);

   glp_add_cols((glp_prob*)linearproblem,1+numkernels);

	//set up theta
   glp_set_col_bnds((glp_prob*)linearproblem,1,GLP_FR,0.0,0.0);
   glp_set_obj_coef((glp_prob*)linearproblem,1,1.0);

	//set up betas
	int32_t offset=2;
	for (int32_t i=0; i<numkernels;++i)
	{
      glp_set_col_bnds((glp_prob*)linearproblem,offset+i,GLP_DB,0.0,1.0);
      glp_set_obj_coef((glp_prob*)linearproblem,offset+i,0.0);
	}

	//set sumupconstraint32_t/sum_l \beta_l=1
   glp_add_rows((glp_prob*)linearproblem,1);

	int32_t*betainds(NULL);
   betainds=SG_MALLOC(int, 1+numkernels);
	for (int32_t i=0; i<numkernels;++i)
	{
		betainds[1+i]=2+i; // coefficient for theta stays zero, therefore
							//start at 2 not at 1 !
	}

	float64_t *betacoeffs(NULL);
	betacoeffs=new float64_t[1+numkernels];

	for (int32_t i=0; i<numkernels;++i)
	{
		betacoeffs[1+i]=1;
	}

   glp_set_mat_row((glp_prob*)linearproblem,1,numkernels, betainds,betacoeffs);
   glp_set_row_bnds((glp_prob*)linearproblem,1,GLP_FX,1.0,1.0);

   SG_FREE(betainds);
   betainds=NULL;

   SG_FREE(betacoeffs);
   betacoeffs=NULL;
#else
	SG_ERROR(
			"glpk.h from GNU glpk not included at compile time necessary "
			"here\n");
#endif

}

void MKLMulticlassGLPK::addconstraint(const ::std::vector<float64_t> & normw2,
		const float64_t sumofpositivealphas)
{
#if defined(USE_GLPK)

	ASSERT ((int)normw2.size()==numkernels)
	ASSERT (sumofpositivealphas>=0)

   glp_add_rows((glp_prob*)linearproblem,1);

   int32_t curconstraint=glp_get_num_rows((glp_prob*)linearproblem);

	int32_t *betainds(NULL);
   betainds=SG_MALLOC(int, 1+1+numkernels);

	betainds[1]=1;
	for (int32_t i=0; i<numkernels;++i)
	{
		betainds[2+i]=2+i; // coefficient for theta stays zero, therefore start
			//at 2 not at 1 !
	}

	float64_t *betacoeffs(NULL);
	betacoeffs=new float64_t[1+1+numkernels];

	betacoeffs[1]=-1;

	for (int32_t i=0; i<numkernels;++i)
	{
		betacoeffs[2+i]=0.5*normw2[i];
	}
   glp_set_mat_row((glp_prob*)linearproblem,curconstraint,1+numkernels, betainds,
			betacoeffs);
   glp_set_row_bnds((glp_prob*)linearproblem,curconstraint,GLP_LO,sumofpositivealphas,
			sumofpositivealphas);

   SG_FREE(betainds);
   betainds=NULL;

   SG_FREE(betacoeffs);
   betacoeffs=NULL;

#else
	SG_ERROR(
			"glpk.h from GNU glpk not included at compile time necessary "
			"here\n");
#endif
}

void MKLMulticlassGLPK::computeweights(std::vector<float64_t> & weights2)
{
#if defined(USE_GLPK)
	weights2.resize(numkernels);

   glp_simplex((glp_prob*) linearproblem,NULL);

	float64_t sum=0;
	for (int32_t i=0; i< numkernels;++i)
	{
      weights2[i]=glp_get_col_prim((glp_prob*) linearproblem, i+2);
		weights2[i]= ::std::max(0.0, ::std::min(1.0,weights2[i]));
		sum+= weights2[i];
	}

	if (sum>0)
	{
		for (int32_t i=0; i< numkernels;++i)
		{
			weights2[i]/=sum;
		}
	}
	else
	SG_ERROR("void glpkwrapper::computeweights(std::vector<float64_t> & "
			"weights2): sum of weights nonpositive %f\n",sum);
#else
	SG_ERROR(
			"glpk.h from GNU glpk not included at compile time necessary "
			"here\n");
#endif
}
