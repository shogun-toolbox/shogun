/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Alexander Binder
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "classifier/svm/gmnpmkl.h"

lpwrapper::lpwrapper()
{
	lpwrappertype=-1;
}

lpwrapper::~lpwrapper()
{

}

void lpwrapper::setup(const int32_t numkernels)
{
	throw ShogunException("void lpwrapper::setup(...): not implemented in derived class");
}

void lpwrapper::addconstraint(const ::std::vector<float64_t> & normw2,
		const float64_t sumofpositivealphas)
{
	throw ShogunException("void lpwrapper::addconstraint(...): not implemented in derived class");

}

void lpwrapper::computeweights(std::vector<float64_t> & weights2)
{
	throw ShogunException("void lpwrapper::computeweights(...): not implemented in derived class");

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
	if (linearproblem!=NULL)
	{
		glp_delete_prob(linearproblem);
		linearproblem=NULL;
	}
	printf("deleting glpk linprob struct\n");

	#endif
}

glpkwrapper glpkwrapper::operator=(glpkwrapper & gl)
{
	throw ShogunException(" glpkwrapper glpkwrapper::operator=(...): must not be called, glpk structure is currently not copiable");

}
glpkwrapper::glpkwrapper(glpkwrapper & gl)
{
	throw ShogunException(" glpkwrapper::glpkwrapper(glpkwrapper & gl): must not be called, glpk structure is currently not copiable");

}

glpkwrapper4CGMNPMKL::glpkwrapper4CGMNPMKL()
{
	numkernels=0;
}

glpkwrapper4CGMNPMKL::~glpkwrapper4CGMNPMKL()
{

}
//TODO: check for correctness of
void glpkwrapper4CGMNPMKL::setup(const int32_t numkernels2)
{
#if defined(USE_GLPK)
	numkernels=numkernels2;
	if(numkernels<=1)
	{
		//std::ostringstream helper;
		//helper << "void glpkwrapper::setup(const int32_tnumkernels): input numkernels out of bounds: "<<numkernels <<std::endl;
		char bla[1000];
		sprintf(bla,"void glpkwrapper::setup(const int32_tnumkernels): input numkernels out of bounds: %d\n",numkernels);
		throw ShogunException(bla);
	}

	if(NULL==linearproblem)
	{
		linearproblem=glp_create_prob();
	}

	glp_set_obj_dir(linearproblem, GLP_MAX);

	glp_add_cols(linearproblem,1+numkernels); // one for theta (objectivelike), all others for betas=weights

	//set up theta
	glp_set_col_name(linearproblem,1,"theta");
	glp_set_col_bnds(linearproblem,1,GLP_FR,0.0,0.0);
	glp_set_obj_coef(linearproblem,1,1.0);

	//set up betas
	int32_t offset=2;
	for(int32_t i=0; i<numkernels;++i)
	{
		//std::ostringstream helper;
		//helper << "beta_"<<i;
		char bla[100];
		sprintf(bla,"beta_%d",i);
		glp_set_col_name(linearproblem,offset+i,bla);
		glp_set_col_bnds(linearproblem,offset+i,GLP_DB,0.0,1.0);
		glp_set_obj_coef(linearproblem,offset+i,0.0);

	}

	// objective is maximize theta over { beta and theta } subject to constraints

	//set sumupconstraint32_t/sum_l \beta_l=1
	glp_add_rows(linearproblem,1);
	glp_set_row_name(linearproblem,1,"betas_sumupto_one");

	int32_t*betainds(NULL);
	betainds=new int[1+numkernels];
	for(int32_t i=0; i<numkernels;++i)
	{
		betainds[1+i]=2+i; // coefficient for theta stays zero, therefore start at 2 not at 1 !
	}

	float64_t *betacoeffs(NULL);
	betacoeffs=new float64_t[1+numkernels];

	for(int32_t i=0; i<numkernels;++i)
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
	SG_ERROR("glpk.h from GNU glpk not included at compile time necessary in gmnpmkl.cpp/.h");
#endif

}

void glpkwrapper4CGMNPMKL::addconstraint(const ::std::vector<float64_t> & normw2,
			const float64_t sumofpositivealphas)
{
#if defined(USE_GLPK)

	ASSERT((int)normw2.size()==numkernels);
	ASSERT(sumofpositivealphas>=0);

	glp_add_rows(linearproblem,1);

	int32_t curconstraint=glp_get_num_rows(linearproblem);

	//std::ostringstream helper;
	//helper << "constraintno_"<<curconstraint;
	char bla[100];
	sprintf(bla,"constraintno_%d",curconstraint);
	glp_set_row_name(linearproblem,curconstraint,bla);

	int32_t *betainds(NULL);
	betainds=new int[1+1+numkernels];

	betainds[1]=1;
	for(int32_t i=0; i<numkernels;++i)
	{
		betainds[2+i]=2+i; // coefficient for theta stays zero, therefore start at 2 not at 1 !
	}

	float64_t *betacoeffs(NULL);
	betacoeffs=new float64_t[1+1+numkernels];

	betacoeffs[1]=-1;


	for(int32_t i=0; i<numkernels;++i)
	{
		betacoeffs[2+i]=0.5*normw2[i];
	}
	glp_set_mat_row(linearproblem,curconstraint,1+numkernels, betainds,betacoeffs);
	glp_set_row_bnds(linearproblem,curconstraint,GLP_LO,sumofpositivealphas,sumofpositivealphas);

	delete[] betainds;
	betainds=NULL;

	delete[] betacoeffs;
	betacoeffs=NULL;

	//addedconstraint=true;
#else
	SG_ERROR("glpk.h from GNU glpk not included at compile time necessary in gmnpmkl.cpp/.h");
#endif
}

void glpkwrapper4CGMNPMKL::computeweights(std::vector<float64_t> & weights2)
{
#if defined(USE_GLPK)
	weights2.resize(numkernels);

	glp_simplex(linearproblem,NULL); // standard parameters

	float64_t sum=0;
	for(int32_t i=0; i< numkernels;++i)
	{
		weights2[i]=glp_get_col_prim(linearproblem, i+2);
		weights2[i]=  ::std::max(0.0, ::std::min(1.0,weights2[i]));
		sum+= weights2[i];
		//
	}

	if(sum>0)
	{
	for(int32_t i=0; i< numkernels;++i)
	{
		 weights2[i]/=sum;
	}
	}
	else
	{
		//std::ostringstream helper;
		//helper << "void glpkwrapper::computeweights(std::vector<float64_t> & weights2): sum of weights nonpositive "<<sum <<std::endl;
		//throw ShogunException(helper.str().c_str());
		char bla[1000];
		sprintf(bla,"void glpkwrapper::computeweights(std::vector<float64_t> & weights2): sum of weights nonpositive %f\n",sum);
		throw ShogunException(bla);
	}



#else
	SG_ERROR("glpk.h from GNU glpk not included at compile time necessary in gmnpmkl.cpp/.h");
#endif
}


CGMNPMKL::CGMNPMKL()
: CMultiClassSVM(ONE_VS_REST)
{
	svm=NULL;
	lpw=NULL;

	lpwrappertype=0;
	thresh=0.01;
	maxiters=999;

	numdat=0;
	numcl=0;
	numker=0;
}

CGMNPMKL::CGMNPMKL(float64_t C, CKernel* k, CLabels* lab)
: CMultiClassSVM(ONE_VS_REST, C, k, lab)
{
	svm=NULL;
	lpw=NULL;

	lpwrappertype=0;
	thresh=0.01;
	maxiters=999;

}


CGMNPMKL::~CGMNPMKL()
{
	SG_UNREF(svm);
	svm=NULL;
	delete lpw;
	lpw=NULL;
}

void CGMNPMKL::lpsetup(const int32_t numkernels)
{
	numker=numkernels;
	ASSERT(numker>0);
	switch(lpwrappertype)
	{
		case 0:
			if(lpw!=NULL)
			{
				delete lpw;
			}
			lpw=new glpkwrapper4CGMNPMKL;
			lpw->setup(numkernels);
		break;

		default:
		{
			//std::ostringstream helper;
			//helper << "CGMNPMKL::setup(const int32_tnumkernels): unknown value for lpwrappertype "<<lpwrappertype <<std::endl;
			//throw ShogunException(helper.str().c_str());

			char bla[1000];
			sprintf(bla,"CGMNPMKL::setup(const int32_tnumkernels): unknown value for lpwrappertype %d\n ",lpwrappertype);
			throw ShogunException(bla);
		}
		break;
	}
}

void CGMNPMKL::initsvm()
{
	ASSERT(labels);

	SG_UNREF(svm);
	svm=new CGMNPSVM;
	SG_REF(svm);

	svm->set_C(get_C1(),get_C2());
	svm->set_epsilon(epsilon);

	//CGNMPSVM->set_labels(get_labels());

	int32_t numlabels;
	float64_t * lb=labels->get_labels ( numlabels);
	ASSERT(numlabels>0);
	numdat=numlabels;
	numcl=labels->get_num_classes();

	CLabels *newlab(NULL);

	newlab=new CLabels(lb, labels->get_num_labels() );
	delete[] lb;
	lb=NULL;

	svm->set_labels(newlab);

	newlab=NULL;

	//TODO test whether labels have been set

}

void CGMNPMKL::init()
{

	if(NULL==kernel)
	{

		throw ShogunException("CGMNPMKL::init(): the set kernel is NULL ");
	}
	else
	{
		if(kernel->get_kernel_type()!=K_COMBINED)
		{
			//std::ostringstream helper;
			//helper << "CGMNPMKL::init(): given kernel is not of type K_COMBINED "<<k->get_kernel_type() <<std::endl;
			//throw ShogunException(helper.str().c_str());

			char bla[1000];
			sprintf(bla,"CGMNPMKL::init(): given kernel is not of type K_COMBINED %d \n",kernel->get_kernel_type());
			throw ShogunException(bla);
		}

		numker=dynamic_cast<CCombinedKernel *>(kernel)->get_num_subkernels();

		lpsetup(numker);


	}
}


bool CGMNPMKL::evaluatefinishcriterion(const int32_t numberofsilpiterations)
{
	if((maxiters>0)&&(numberofsilpiterations>=maxiters))
	{
		return(true);
	}

	if(weightshistory.size()>1)
	{
		std::vector<float64_t> wold,wnew;

		wold=weightshistory[ weightshistory.size()-2 ];
		wnew=weightshistory.back();
		float64_t delta=0;

		ASSERT(wold.size()==wnew.size());

		for(size_t i=0;i< wnew.size();++i)
		{
			delta+=(wold[i]-wnew[i])*(wold[i]-wnew[i]);
		}
		delta=sqrt(delta);

		SG_SPRINT( "CGMNPMKL::evaluatefinishcriterion(): L2 norm of changes= %f, required for termination by member variables thresh= %f \n",delta, thresh);

		if( (delta < thresh)&&(numberofsilpiterations>=1) )
		{
			return(true);
		}
	}

	return(false);
}

void CGMNPMKL::addingweightsstep( const std::vector<float64_t> & curweights)
{
	//weightshistory.push_back(curweights);
	//error prone?
	if(weightshistory.size()>2)
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

	//number of labels equal to number of features?
	ASSERT(numdat==kernel->get_num_vec_lhs());
	ASSERT(numdat==kernel->get_num_vec_rhs());

	svm->set_kernel(kernel);
	svm->train();

	float64_t sumofsignfreealphas=getsumofsignfreealphas();
	int32_t numkernels=dynamic_cast<CCombinedKernel *>(kernel)->get_num_subkernels();


	std::vector<float64_t> normw2(numkernels);
	for(int32_t ind=0; ind < numkernels; ++ind )
	{
		normw2[ind]=getsquarenormofprimalcoefficients( ind );
	}

	//add a constraint
	lpw->addconstraint(normw2,sumofsignfreealphas);
}

float64_t CGMNPMKL::getsumofsignfreealphas()
{
	//returns \sum_y b_y^2-\sum_i \sum_{ y \neq y_i} \alpha_{iy}(b_{y_i}-b_y-1)

	std::vector<int> trainlabels2(labels->get_num_labels());
	int32_t tmpint;
	int32_t * lab=labels->get_int_labels ( tmpint);
	std::copy(lab,lab+labels->get_num_labels(), trainlabels2.begin());
	delete[] lab;
	lab=NULL;


	ASSERT(trainlabels2.size()>0);
	float64_t sum=0;

	for(int32_t nc=0; nc< labels->get_num_classes();++nc)
	{
		CSVM * sm=svm->get_svm(nc);

		float64_t bia=sm->get_bias();
		sum+= bia*bia;

		SG_UNREF(sm);
	}

	::std::vector< ::std::vector<float64_t> > basealphas;
	svm->getbasealphas( basealphas);

	for(size_t lb=0; lb< trainlabels2.size();++lb)
	{
		for(int32_t nc=0; nc< labels->get_num_classes();++nc)
		{

			CSVM * sm=svm->get_svm(nc);


			if((int)nc!=trainlabels2[lb])
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

float64_t CGMNPMKL::getsquarenormofprimalcoefficients(
		const int32_t ind)
{
	// alphas are already correctly transformed!

	CKernel * ker=dynamic_cast<CCombinedKernel *>(kernel)->get_kernel(ind);

	float64_t tmp=0;

	for(int32_t classindex=0; classindex< labels->get_num_classes();++classindex)
	{
		CSVM * sm=svm->get_svm(classindex);

		for (int32_t i=0; i < sm->get_num_support_vectors(); ++i)
		{
			float64_t alphai=sm->get_alpha(i);// svmstuff[classindex].alphas[i];
			int32_t svindi= sm->get_support_vector(i); //svmstuff[classindex].svind[i];

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


bool CGMNPMKL::train()
{
	init();
	weightshistory.clear();

	int32_t numkernels=dynamic_cast<CCombinedKernel *>(kernel)->get_num_subkernels();

	::std::vector<float64_t> curweights(numkernels,1.0/numkernels);
	weightshistory.push_back(curweights);

	SG_PRINT("initial weights in silp \n");
	for(size_t i=0; i< curweights.size();++i)
	{
		SG_PRINT("%f ",curweights[i]);
	}
	SG_PRINT("\n");


	addingweightsstep(curweights);


	int32_t numberofsilpiterations=0;
		bool final=false;
		while(false==final)
		{

			curweights.clear();
			lpw->computeweights(curweights);
			weightshistory.push_back(curweights);

			SG_SPRINT("SILP iteration %d weights in silp \n",numberofsilpiterations);
			for(size_t i=0; i< curweights.size();++i)
			{
				SG_SPRINT("%f ",curweights[i]);
			}
			SG_SPRINT("\n");

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
			SG_REF(nsvm);
			for (int32_t k=0; k<osvm->get_num_support_vectors() ; k++)
			{
				nsvm->set_alpha(k, osvm->get_alpha(k) );
				nsvm->set_support_vector(k,osvm->get_support_vector(k) );
			}
			nsvm->set_bias(osvm->get_bias() );
			SG_UNREF(osvm);
			osvm=NULL;
			set_svm(i, nsvm);
		}
		SG_UNREF(svm);
		svm=NULL;
		return(true);
}




float64_t* CGMNPMKL::getsubkernelweights(int32_t & numweights)
{
	if((numker<=0 )||weightshistory.empty() )
	{
		numweights=0;
		return(NULL);
	}

	std::vector<float64_t> subkerw=weightshistory.back();
	ASSERT(numker=subkerw.size());
	numweights=numker;

	float64_t* res=new  float64_t[numker];
	std::copy(subkerw.begin(), subkerw.end(),res);
	return(res);
}

