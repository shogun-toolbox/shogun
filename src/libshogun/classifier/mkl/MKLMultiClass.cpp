/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Alexander Binder
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/mkl/MKLMultiClass.h"
#include "lib/io.h"

using namespace shogun;

lpwrapper::lpwrapper()
{
	lpwrappertype=-1;
}

lpwrapper::~lpwrapper()
{

}

void lpwrapper::setup(const int32_t numkernels)
{
	SG_ERROR("void lpwrapper::setup(...): not implemented in derived class\n");
}

void lpwrapper::addconstraint(const ::std::vector<float64_t> & normw2,
		const float64_t sumofpositivealphas)
{
	SG_ERROR("void lpwrapper::addconstraint(...): not implemented in derived class");
}

void lpwrapper::computeweights(std::vector<float64_t> & weights2)
{
	SG_ERROR("void lpwrapper::computeweights(...): not implemented in derived class\n");
}

// ***************************************************************************

glpkwrapper::glpkwrapper()
{
	lpwrappertype=0;

#if defined(USE_GLPK)
	linearproblem=NULL;
#endif
}
glpkwrapper::~glpkwrapper()
{
#if defined(USE_GLPK)
	if (linearproblem)
	{
		glp_delete_prob(linearproblem);
		linearproblem=NULL;
	}
	printf("deleting glpk linprob struct\n");

	#endif
}

glpkwrapper glpkwrapper::operator=(glpkwrapper & gl)
{
	SG_ERROR(" glpkwrapper glpkwrapper::operator=(...): must not be called, glpk structure is currently not copiable");
	return(*this);

}
glpkwrapper::glpkwrapper(glpkwrapper & gl)
{
	SG_ERROR(" glpkwrapper::glpkwrapper(glpkwrapper & gl): must not be called, glpk structure is currently not copiable");

}

glpkwrapper4CGMNPMKL::glpkwrapper4CGMNPMKL()
{
	numkernels=0;
}

glpkwrapper4CGMNPMKL::~glpkwrapper4CGMNPMKL()
{

}

void glpkwrapper4CGMNPMKL::setup(const int32_t numkernels2)
{
#if defined(USE_GLPK)
	numkernels=numkernels2;
	if (numkernels<=1)
	{
		SG_ERROR("void glpkwrapper::setup(const int32_tnumkernels): input numkernels out of bounds: %d\n",numkernels);
	}

	if (!linearproblem)
	{
		linearproblem=glp_create_prob();
	}

	glp_set_obj_dir(linearproblem, GLP_MAX);

	glp_add_cols(linearproblem,1+numkernels); // one for theta (objective), all others for betas=weights

	//set up theta
	glp_set_col_bnds(linearproblem,1,GLP_FR,0.0,0.0);
	glp_set_obj_coef(linearproblem,1,1.0);

	//set up betas
	int32_t offset=2;
	for (int32_t i=0; i<numkernels;++i)
	{
		glp_set_col_bnds(linearproblem,offset+i,GLP_DB,0.0,1.0);
		glp_set_obj_coef(linearproblem,offset+i,0.0);
	}

	//set sumupconstraint32_t/sum_l \beta_l=1
	glp_add_rows(linearproblem,1);

	int32_t*betainds(NULL);
	betainds=new int[1+numkernels];
	for (int32_t i=0; i<numkernels;++i)
	{
		betainds[1+i]=2+i; // coefficient for theta stays zero, therefore start at 2 not at 1 !
	}

	float64_t *betacoeffs(NULL);
	betacoeffs=new float64_t[1+numkernels];

	for (int32_t i=0; i<numkernels;++i)
	{
		betacoeffs[1+i]=1;
	}

	glp_set_mat_row(linearproblem,1,numkernels, betainds,betacoeffs);
	glp_set_row_bnds(linearproblem,1,GLP_FX,1.0,1.0);

	delete[] betainds;
	betainds=NULL;

	delete[] betacoeffs;
	betacoeffs=NULL;
#else
	SG_ERROR("glpk.h from GNU glpk not included at compile time necessary here");
#endif

}

void glpkwrapper4CGMNPMKL::addconstraint(const ::std::vector<float64_t> & normw2,
			const float64_t sumofpositivealphas)
{
#if defined(USE_GLPK)

	ASSERT ((int)normw2.size()==numkernels);
	ASSERT (sumofpositivealphas>=0);

	glp_add_rows(linearproblem,1);

	int32_t curconstraint=glp_get_num_rows(linearproblem);

	int32_t *betainds(NULL);
	betainds=new int[1+1+numkernels];

	betainds[1]=1;
	for (int32_t i=0; i<numkernels;++i)
	{
		betainds[2+i]=2+i; // coefficient for theta stays zero, therefore start at 2 not at 1 !
	}

	float64_t *betacoeffs(NULL);
	betacoeffs=new float64_t[1+1+numkernels];

	betacoeffs[1]=-1;

	for (int32_t i=0; i<numkernels;++i)
	{
		betacoeffs[2+i]=0.5*normw2[i];
	}
	glp_set_mat_row(linearproblem,curconstraint,1+numkernels, betainds,betacoeffs);
	glp_set_row_bnds(linearproblem,curconstraint,GLP_LO,sumofpositivealphas,sumofpositivealphas);

	delete[] betainds;
	betainds=NULL;

	delete[] betacoeffs;
	betacoeffs=NULL;

#else
	SG_ERROR("glpk.h from GNU glpk not included at compile time necessary in gmnpmkl.cpp/.h");
#endif
}

void glpkwrapper4CGMNPMKL::computeweights(std::vector<float64_t> & weights2)
{
#if defined(USE_GLPK)
	weights2.resize(numkernels);

	glp_simplex(linearproblem,NULL); 

	float64_t sum=0;
	for (int32_t i=0; i< numkernels;++i)
	{
		weights2[i]=glp_get_col_prim(linearproblem, i+2);
		weights2[i]=  ::std::max(0.0, ::std::min(1.0,weights2[i]));
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
		SG_ERROR("void glpkwrapper::computeweights(std::vector<float64_t> & weights2): sum of weights nonpositive %f\n",sum);
#else
	SG_ERROR("glpk.h from GNU glpk not included at compile time necessary here");
#endif
}


CMKLMultiClass::CMKLMultiClass()
: CMultiClassSVM(ONE_VS_REST)
{
	svm=NULL;
	lpw=NULL;

	mkl_eps=0.01;
	max_num_mkl_iters=999;
}

CMKLMultiClass::CMKLMultiClass(float64_t C, CKernel* k, CLabels* lab)
: CMultiClassSVM(ONE_VS_REST, C, k, lab)
{
	svm=NULL;
	lpw=NULL;
	
	mkl_eps=0.01;
	max_num_mkl_iters=999;

}


CMKLMultiClass::~CMKLMultiClass()
{
	SG_UNREF(svm);
	svm=NULL;
	delete lpw;
	lpw=NULL;
}


void CMKLMultiClass::initsvm()
{
	if (!labels)	
	{
		SG_ERROR("CMKLMultiClass::initsvm(): the set labels is NULL\n");
	}

	SG_UNREF(svm);
	svm=new CGMNPSVM;
	SG_REF(svm);

	svm->set_C(get_C1(),get_C2());
	svm->set_epsilon(epsilon);

	int32_t numlabels;
	float64_t * lb=labels->get_labels ( numlabels);

	if (numlabels<=0)	
	{
		SG_ERROR("CMKLMultiClass::initsvm(): the number of labels is nonpositive, do not know how to handle this!\n");
	}

	CLabels* newlab=new CLabels(lb, labels->get_num_labels() );
	delete[] lb;
	lb=NULL;

	svm->set_labels(newlab);

	newlab=NULL;
}

void CMKLMultiClass::initlpsolver()
{
	if (!kernel)	
	{
		SG_ERROR("CMKLMultiClass::initlpsolver(): the set kernel is NULL\n");
	}

	if (kernel->get_kernel_type()!=K_COMBINED)
	{
		SG_ERROR("CMKLMultiClass::initlpsolver(): given kernel is not of type K_COMBINED %d required by Multiclass Mkl \n",kernel->get_kernel_type());
	}

	int numker=dynamic_cast<CCombinedKernel *>(kernel)->get_num_subkernels();

	ASSERT(numker>0);

	if (lpw)
	{
		delete lpw;
	}
	lpw=new glpkwrapper4CGMNPMKL;
	lpw->setup(numker);
}


bool CMKLMultiClass::evaluatefinishcriterion(const int32_t numberofsilpiterations)
{
	if ( (max_num_mkl_iters>0) && (numberofsilpiterations>=max_num_mkl_iters) )
	{
		return(true);
	}

	if (weightshistory.size()>1)
	{
		std::vector<float64_t> wold,wnew;

		wold=weightshistory[ weightshistory.size()-2 ];
		wnew=weightshistory.back();
		float64_t delta=0;

		ASSERT (wold.size()==wnew.size());

		for (size_t i=0;i< wnew.size();++i)
		{
			delta+=(wold[i]-wnew[i])*(wold[i]-wnew[i]);
		}
		delta=sqrt(delta);

		if( (delta < mkl_eps) && (numberofsilpiterations>=1) )
		{
			return(true);
		}
	}

	return(false);
}

void CMKLMultiClass::addingweightsstep( const std::vector<float64_t> & curweights)
{

	if (weightshistory.size()>2)
	{
		weightshistory.erase(weightshistory.begin());
	}

	float64_t* weights(NULL);
	weights=new float64_t[curweights.size()];
	std::copy(curweights.begin(),curweights.end(),weights);

	kernel->set_subkernel_weights(  weights, curweights.size());
	delete[] weights;
	weights=NULL;

	initsvm();

	svm->set_kernel(kernel);
	svm->train();

	float64_t sumofsignfreealphas=getsumofsignfreealphas();
	int32_t numkernels=dynamic_cast<CCombinedKernel *>(kernel)->get_num_subkernels();


	std::vector<float64_t> normw2(numkernels);
	for (int32_t ind=0; ind < numkernels; ++ind )
	{
		normw2[ind]=getsquarenormofprimalcoefficients( ind );
	}

	lpw->addconstraint(normw2,sumofsignfreealphas);
}

float64_t CMKLMultiClass::getsumofsignfreealphas()
{

	std::vector<int> trainlabels2(labels->get_num_labels());
	int32_t tmpint;
	int32_t * lab=labels->get_int_labels ( tmpint);
	std::copy(lab,lab+labels->get_num_labels(), trainlabels2.begin());
	delete[] lab;
	lab=NULL;


	ASSERT (trainlabels2.size()>0);
	float64_t sum=0;

	for (int32_t nc=0; nc< labels->get_num_classes();++nc)
	{
		CSVM * sm=svm->get_svm(nc);

		float64_t bia=sm->get_bias();
		sum+= bia*bia;

		SG_UNREF(sm);
	}

	::std::vector< ::std::vector<float64_t> > basealphas;
	svm->getbasealphas( basealphas);

	for (size_t lb=0; lb< trainlabels2.size();++lb)
	{
		for (int32_t nc=0; nc< labels->get_num_classes();++nc)
		{
			CSVM * sm=svm->get_svm(nc);

			if ((int)nc!=trainlabels2[lb])
			{
				CSVM * sm2=svm->get_svm(trainlabels2[lb]);

				float64_t bia1=sm2->get_bias();
				float64_t bia2=sm->get_bias();
				SG_UNREF(sm2);

				sum+= -basealphas[nc][lb]*(bia1-bia2-1);
			}
			SG_UNREF(sm);
		}
	}

	return(sum);
}

float64_t CMKLMultiClass::getsquarenormofprimalcoefficients(
		const int32_t ind)
{
	CKernel * ker=dynamic_cast<CCombinedKernel *>(kernel)->get_kernel(ind);

	float64_t tmp=0;

	for (int32_t classindex=0; classindex< labels->get_num_classes();++classindex)
	{
		CSVM * sm=svm->get_svm(classindex);

		for (int32_t i=0; i < sm->get_num_support_vectors(); ++i)
		{
			float64_t alphai=sm->get_alpha(i);
			int32_t svindi= sm->get_support_vector(i); 

			for (int32_t k=0; k < sm->get_num_support_vectors(); ++k)
			{
				float64_t alphak=sm->get_alpha(k);
				int32_t svindk=sm->get_support_vector(k);

				tmp+=alphai*ker->kernel(svindi,svindk)
				*alphak;

			}
		}
		SG_UNREF(sm);
	}
	SG_UNREF(ker);
	ker=NULL;

	return(tmp);
}


bool CMKLMultiClass::train(CFeatures* data)
{
	//makes glpk quiet
	glp_term_out(GLP_OFF);

	int numcl=labels->get_num_classes();
	ASSERT(kernel);
	ASSERT(labels && labels->get_num_labels());

	if (data)
	{
		if (labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n");
		kernel->init(data, data);
	}

	initlpsolver();
	weightshistory.clear();

	int32_t numkernels=dynamic_cast<CCombinedKernel *>(kernel)->get_num_subkernels();

	::std::vector<float64_t> curweights(numkernels,1.0/numkernels);
	weightshistory.push_back(curweights);

	addingweightsstep(curweights);

	int32_t numberofsilpiterations=0;
	bool final=false;
	while (!final)
	{

		curweights.clear();
		lpw->computeweights(curweights);
		weightshistory.push_back(curweights);


		final=evaluatefinishcriterion(numberofsilpiterations);
		++numberofsilpiterations;

		addingweightsstep(curweights);

	} // while(false==final)


	//set alphas, bias, support vecs
	ASSERT(numcl>=1);
	create_multiclass_svm(numcl);

	for (int32_t i=0; i<numcl; i++)
	{
		CSVM* osvm=svm->get_svm(i);
		CSVM* nsvm=new CSVM(osvm->get_num_support_vectors());

		for (int32_t k=0; k<osvm->get_num_support_vectors() ; k++)
		{
			nsvm->set_alpha(k, osvm->get_alpha(k) );
			nsvm->set_support_vector(k,osvm->get_support_vector(k) );
		}
		nsvm->set_bias(osvm->get_bias() );
		set_svm(i, nsvm);

		SG_UNREF(osvm);
		osvm=NULL;
	}

	SG_UNREF(svm);
	svm=NULL;
	return(true);
}




float64_t* CMKLMultiClass::getsubkernelweights(int32_t & numweights)
{
	if ( weightshistory.empty() )
	{
		numweights=0;
		return NULL;
	}

	std::vector<float64_t> subkerw=weightshistory.back();
	numweights=weightshistory.back().size();

	float64_t* res=new float64_t[numweights];
	std::copy(weightshistory.back().begin(), weightshistory.back().end(),res);
	return res;
}

void CMKLMultiClass::set_mkl_epsilon(float64_t eps )
{
	mkl_eps=eps;
}

void CMKLMultiClass::set_max_num_mkliters(int32_t maxnum)
{
	max_num_mkl_iters=maxnum;
}
