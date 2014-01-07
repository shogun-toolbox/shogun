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

#include <multiclass/MulticlassOneVsRestStrategy.h>
#include <classifier/mkl/MKLMulticlass.h>
#include <io/SGIO.h>
#include <labels/MulticlassLabels.h>

using namespace shogun;


CMKLMulticlass::CMKLMulticlass()
: CMulticlassSVM(new CMulticlassOneVsRestStrategy())
{
	svm=NULL;
	lpw=NULL;

	mkl_eps=0.01;
	max_num_mkl_iters=999;
	pnorm=1;
}

CMKLMulticlass::CMKLMulticlass(float64_t C, CKernel* k, CLabels* lab)
: CMulticlassSVM(new CMulticlassOneVsRestStrategy(), C, k, lab)
{
	svm=NULL;
	lpw=NULL;

	mkl_eps=0.01;
	max_num_mkl_iters=999;
	pnorm=1;
}


CMKLMulticlass::~CMKLMulticlass()
{
	SG_UNREF(svm);
	svm=NULL;
	delete lpw;
	lpw=NULL;
}

CMKLMulticlass::CMKLMulticlass( const CMKLMulticlass & cm)
: CMulticlassSVM(new CMulticlassOneVsRestStrategy())
{
	svm=NULL;
	lpw=NULL;
	SG_ERROR(
         " CMKLMulticlass::CMKLMulticlass(const CMKLMulticlass & cm): must "
			"not be called, glpk structure is currently not copyable");
}

CMKLMulticlass CMKLMulticlass::operator=( const CMKLMulticlass & cm)
{
		SG_ERROR(
         " CMKLMulticlass CMKLMulticlass::operator=(...): must "
			"not be called, glpk structure is currently not copyable");
	return (*this);
}


void CMKLMulticlass::initsvm()
{
   if (!m_labels)
	{
      SG_ERROR("CMKLMulticlass::initsvm(): the set labels is NULL\n")
	}

	SG_UNREF(svm);
	svm=new CGMNPSVM;
	SG_REF(svm);

   svm->set_C(get_C());
   svm->set_epsilon(get_epsilon());

   if (m_labels->get_num_labels()<=0)
	{
      SG_ERROR("CMKLMulticlass::initsvm(): the number of labels is "
				"nonpositive, do not know how to handle this!\n");
	}

   svm->set_labels(m_labels);
}

void CMKLMulticlass::initlpsolver()
{
   if (!m_kernel)
	{
      SG_ERROR("CMKLMulticlass::initlpsolver(): the set kernel is NULL\n")
	}

   if (m_kernel->get_kernel_type()!=K_COMBINED)
	{
      SG_ERROR("CMKLMulticlass::initlpsolver(): given kernel is not of type"
            " K_COMBINED %d required by Multiclass Mkl \n",
            m_kernel->get_kernel_type());
	}

   int numker=dynamic_cast<CCombinedKernel *>(m_kernel)->get_num_subkernels();

	ASSERT(numker>0)
	/*
	if (lpw)
	{
		delete lpw;
	}
	*/

   //lpw=new MKLMulticlassGLPK;
	if(pnorm>1)
	{
      lpw=new MKLMulticlassGradient;
		lpw->set_mkl_norm(pnorm);
	}
	else
	{
      lpw=new MKLMulticlassGLPK;
	}
	lpw->setup(numker);

}


bool CMKLMulticlass::evaluatefinishcriterion(const int32_t
		numberofsilpiterations)
{
	if ( (max_num_mkl_iters>0) && (numberofsilpiterations>=max_num_mkl_iters) )
		return true;

	if (weightshistory.size()>1)
	{
		std::vector<float64_t> wold,wnew;

		wold=weightshistory[ weightshistory.size()-2 ];
		wnew=weightshistory.back();
		float64_t delta=0;

		ASSERT (wold.size()==wnew.size())


		if((pnorm<=1)&&(!normweightssquared.empty()))
		{
			//check dual gap part for mkl

			delta=oldalphaterm-curalphaterm;

			int32_t maxind=0;
			float64_t maxval=normweightssquared[maxind];
			for (size_t i=0;i< wnew.size();++i)
			{
				delta+=-0.5*oldnormweightssquared[i]*wold[i];
				if(normweightssquared[i]>maxval)
				{
					maxind=i;
					maxval=normweightssquared[i];
				}
			}
			delta+=0.5*normweightssquared[maxind];
			//delta=fabs(delta);
			SG_SINFO("L1 Norm chosen, MKL part of duality gap %f \n",delta)
			if( (delta < mkl_eps) && (numberofsilpiterations>=1) )
			{
				return true;
			}



		}
		else
		{
			delta=0;

			float64_t deltaold=oldalphaterm,deltanew=curalphaterm;
			for (size_t i=0;i< wnew.size();++i)
			{
				delta+=(wold[i]-wnew[i])*(wold[i]-wnew[i]);
				deltaold+= -0.5*oldnormweightssquared[i]*wold[i];
				deltanew+= -0.5*normweightssquared[i]*wnew[i];
			}
			if(deltanew>0)
			{
			delta=1-deltanew/deltaold;
			}
			else
			{
            SG_SWARNING("CMKLMulticlass::evaluatefinishcriterion(...): deltanew<=0.Switching back to weight norsm difference as criterion.\n")
				delta=sqrt(delta);
			}
				SG_SINFO("weight delta %f \n",delta)

			if( (delta < mkl_eps) && (numberofsilpiterations>=1) )
			{
				return true;
			}

		}
	}

	return false;
}

void CMKLMulticlass::addingweightsstep( const std::vector<float64_t> &
		curweights)
{

	if (weightshistory.size()>2)
	{
		weightshistory.erase(weightshistory.begin());
	}

   //float64_t* weights(NULL);
   //weights=new float64_t[curweights.size()];
   SGVector<float64_t> weights(curweights.size());
   std::copy(curweights.begin(),curweights.end(),weights.vector);

   m_kernel->set_subkernel_weights(weights);
   //delete[] weights;
   //weights=NULL;

	initsvm();

   svm->set_kernel(m_kernel);
	svm->train();

	float64_t sumofsignfreealphas=getsumofsignfreealphas();
	curalphaterm=sumofsignfreealphas;

	int32_t numkernels=
         dynamic_cast<CCombinedKernel *>(m_kernel)->get_num_subkernels();


	normweightssquared.resize(numkernels);
	for (int32_t ind=0; ind < numkernels; ++ind )
	{
		normweightssquared[ind]=getsquarenormofprimalcoefficients( ind );
	}

	lpw->addconstraint(normweightssquared,sumofsignfreealphas);
}

float64_t CMKLMulticlass::getsumofsignfreealphas()
{

   std::vector<int> trainlabels2(m_labels->get_num_labels());
   SGVector<int32_t> lab=((CMulticlassLabels*) m_labels)->get_int_labels();
   std::copy(lab.vector,lab.vector+lab.vlen, trainlabels2.begin());

	ASSERT (trainlabels2.size()>0)
	float64_t sum=0;

   for (int32_t nc=0; nc< ((CMulticlassLabels*) m_labels)->get_num_classes();++nc)
	{
		CSVM * sm=svm->get_svm(nc);

		float64_t bia=sm->get_bias();
		sum+= 0.5*bia*bia;

		SG_UNREF(sm);
	}

	index_t basealphas_y = 0, basealphas_x = 0;
	float64_t* basealphas = svm->get_basealphas_ptr(&basealphas_y,
													&basealphas_x);

	for (size_t lb=0; lb< trainlabels2.size();++lb)
	{
      for (int32_t nc=0; nc< ((CMulticlassLabels*) m_labels)->get_num_classes();++nc)
		{
			CSVM * sm=svm->get_svm(nc);

			if ((int)nc!=trainlabels2[lb])
			{
				CSVM * sm2=svm->get_svm(trainlabels2[lb]);

				float64_t bia1=sm2->get_bias();
				float64_t bia2=sm->get_bias();
				SG_UNREF(sm2);

				sum+= -basealphas[lb*basealphas_y + nc]*(bia1-bia2-1);
			}
			SG_UNREF(sm);
		}
	}

	return sum;
}

float64_t CMKLMulticlass::getsquarenormofprimalcoefficients(
		const int32_t ind)
{
   CKernel * ker=dynamic_cast<CCombinedKernel *>(m_kernel)->get_kernel(ind);

	float64_t tmp=0;

   for (int32_t classindex=0; classindex< ((CMulticlassLabels*) m_labels)->get_num_classes();
			++classindex)
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

	return tmp;
}


bool CMKLMulticlass::train_machine(CFeatures* data)
{
   ASSERT(m_kernel)
   ASSERT(m_labels && m_labels->get_num_labels())
   ASSERT(m_labels->get_label_type() == LT_MULTICLASS)

   int numcl=((CMulticlassLabels*) m_labels)->get_num_classes();

	if (data)
	{
      if (m_labels->get_num_labels() != data->get_num_vectors())
      {
         SG_ERROR("%s::train_machine(): Number of training vectors (%d) does"
               " not match number of labels (%d)\n", get_name(),
               data->get_num_vectors(), m_labels->get_num_labels());
      }
      m_kernel->init(data, data);
	}

	initlpsolver();

	weightshistory.clear();

	int32_t numkernels=
         dynamic_cast<CCombinedKernel *>(m_kernel)->get_num_subkernels();

	::std::vector<float64_t> curweights(numkernels,1.0/numkernels);
	weightshistory.push_back(curweights);

	addingweightsstep(curweights);

	oldalphaterm=curalphaterm;
	oldnormweightssquared=normweightssquared;

	int32_t numberofsilpiterations=0;
	bool final=false;
	while (!final)
	{

		//curweights.clear();
		lpw->computeweights(curweights);
		weightshistory.push_back(curweights);

		addingweightsstep(curweights);

		//new weights new biasterm

		final=evaluatefinishcriterion(numberofsilpiterations);

		oldalphaterm=curalphaterm;
		oldnormweightssquared=normweightssquared;

		++numberofsilpiterations;


	} // while(false==final)


	//set alphas, bias, support vecs
	ASSERT(numcl>=1)
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
	if (lpw)
	{
		delete lpw;
	}
	lpw=NULL;
	return true;
}




float64_t* CMKLMulticlass::getsubkernelweights(int32_t & numweights)
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

void CMKLMulticlass::set_mkl_epsilon(float64_t eps )
{
	mkl_eps=eps;
}

void CMKLMulticlass::set_max_num_mkliters(int32_t maxnum)
{
	max_num_mkl_iters=maxnum;
}

void CMKLMulticlass::set_mkl_norm(float64_t norm)
{
	pnorm=norm;
	if(pnorm<1 )
      SG_ERROR("CMKLMulticlass::set_mkl_norm(float64_t norm) : parameter pnorm<1")
}
