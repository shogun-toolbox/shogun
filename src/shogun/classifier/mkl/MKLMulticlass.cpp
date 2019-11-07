/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Chiyuan Zhang, Giovanni De Toni,
 *          Heiko Strathmann, Sergey Lisitsyn, Bjoern Esser, Saurabh Goyal
 */

#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/classifier/mkl/MKLMulticlass.h>
#include <shogun/io/SGIO.h>
#include <shogun/labels/MulticlassLabels.h>

#include <shogun/lib/Signal.h>
#include <utility>
#include <vector>

using namespace shogun;


MKLMulticlass::MKLMulticlass()
: MulticlassSVM(std::make_shared<MulticlassOneVsRestStrategy>())
{
	svm=NULL;
	lpw=NULL;

	mkl_eps=0.01;
	max_num_mkl_iters=999;
	pnorm=1;
}

MKLMulticlass::MKLMulticlass(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab)
: MulticlassSVM(std::make_shared<MulticlassOneVsRestStrategy>(), C, std::move(k), std::move(lab))
{
	svm=NULL;
	lpw=NULL;

	mkl_eps=0.01;
	max_num_mkl_iters=999;
	pnorm=1;
}


MKLMulticlass::~MKLMulticlass()
{
}

MKLMulticlass::MKLMulticlass( const MKLMulticlass & cm)
: MulticlassSVM(std::make_shared<MulticlassOneVsRestStrategy>())
{
	svm=NULL;
	lpw=NULL;
	error(
         " MKLMulticlass::MKLMulticlass(const MKLMulticlass & cm): must "
			"not be called, glpk structure is currently not copyable");
}

MKLMulticlass MKLMulticlass::operator=( const MKLMulticlass & cm)
{
		error(
         " MKLMulticlass MKLMulticlass::operator=(...): must "
			"not be called, glpk structure is currently not copyable");
	return (*this);
}


void MKLMulticlass::initsvm()
{
   if (!m_labels)
	{
      error("MKLMulticlass::initsvm(): the set labels is NULL");
	}


	svm=std::make_shared<GMNPSVM>();


   svm->set_C(get_C());
   svm->set_epsilon(get_epsilon());

   if (m_labels->get_num_labels()<=0)
	{
      error("MKLMulticlass::initsvm(): the number of labels is "
				"nonpositive, do not know how to handle this!\n");
	}

   svm->set_labels(m_labels);
}

void MKLMulticlass::initlpsolver()
{
   if (!m_kernel)
	{
      error("MKLMulticlass::initlpsolver(): the set kernel is NULL");
	}

   if (m_kernel->get_kernel_type()!=K_COMBINED)
	{
      error("MKLMulticlass::initlpsolver(): given kernel is not of type"
            " K_COMBINED {} required by Multiclass Mkl \n",
            m_kernel->get_kernel_type());
	}

   int numker=std::dynamic_pointer_cast<CombinedKernel>(m_kernel)->get_num_subkernels();

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
    	lpw=std::make_shared<MKLMulticlassGradient>();
		lpw->set_mkl_norm(pnorm);
	}
	else
	{
      lpw=std::make_shared<MKLMulticlassGLPK>();
	}
	lpw->setup(numker);

}


bool MKLMulticlass::evaluatefinishcriterion(const int32_t
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
			io::info("L1 Norm chosen, MKL part of duality gap {} ",delta);
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
            io::warn("MKLMulticlass::evaluatefinishcriterion(...): deltanew<=0.Switching back to weight norsm difference as criterion.");
				delta=sqrt(delta);
			}
				io::info("weight delta {} ",delta);

			if( (delta < mkl_eps) && (numberofsilpiterations>=1) )
			{
				return true;
			}

		}
	}

	return false;
}

void MKLMulticlass::addingweightsstep( const std::vector<float64_t> &
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
         std::dynamic_pointer_cast<CombinedKernel>(m_kernel)->get_num_subkernels();


	normweightssquared.resize(numkernels);
	for (int32_t ind=0; ind < numkernels; ++ind )
	{
		normweightssquared[ind]=getsquarenormofprimalcoefficients( ind );
	}

	lpw->addconstraint(normweightssquared,sumofsignfreealphas);
}

float64_t MKLMulticlass::getsumofsignfreealphas()
{

   std::vector<int> trainlabels2(m_labels->get_num_labels());
   SGVector<int32_t> lab=(std::static_pointer_cast<MulticlassLabels>(m_labels))->get_int_labels();
   std::copy(lab.vector,lab.vector+lab.vlen, trainlabels2.begin());

	ASSERT (trainlabels2.size()>0)
	float64_t sum=0;

   for (int32_t nc=0; nc< (std::static_pointer_cast<MulticlassLabels>(m_labels))->get_num_classes();++nc)
	{
		auto sm=svm->get_svm(nc);

		float64_t bia=sm->get_bias();
		sum+= 0.5*bia*bia;


	}

	index_t basealphas_y = 0, basealphas_x = 0;
	float64_t* basealphas = svm->get_basealphas_ptr(&basealphas_y,
													&basealphas_x);

	for (size_t lb=0; lb< trainlabels2.size();++lb)
	{
      for (int32_t nc=0; nc< (std::static_pointer_cast<MulticlassLabels>(m_labels))->get_num_classes();++nc)
		{
			auto sm=svm->get_svm(nc);

			if ((int)nc!=trainlabels2[lb])
			{
				auto sm2=svm->get_svm(trainlabels2[lb]);

				float64_t bia1=sm2->get_bias();
				float64_t bia2=sm->get_bias();


				sum+= -basealphas[lb*basealphas_y + nc]*(bia1-bia2-1);
			}

		}
	}

	return sum;
}

float64_t MKLMulticlass::getsquarenormofprimalcoefficients(
		const int32_t ind)
{
   auto ker=std::dynamic_pointer_cast<CombinedKernel>(m_kernel)->get_kernel(ind);

	float64_t tmp=0;

   for (int32_t classindex=0; classindex< (std::static_pointer_cast<MulticlassLabels>(m_labels))->get_num_classes();
			++classindex)
	{
		auto sm=svm->get_svm(classindex);

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

	}

	ker=NULL;

	return tmp;
}


bool MKLMulticlass::train_machine(std::shared_ptr<Features> data)
{
   ASSERT(m_kernel)
   ASSERT(m_labels && m_labels->get_num_labels())
   ASSERT(m_labels->get_label_type() == LT_MULTICLASS)
   init_strategy();

   int numcl=(std::static_pointer_cast<MulticlassLabels>(m_labels))->get_num_classes();

	if (data)
	{
      if (m_labels->get_num_labels() != data->get_num_vectors())
      {
         error("{}::train_machine(): Number of training vectors ({}) does"
               " not match number of labels ({})\n", get_name(),
               data->get_num_vectors(), m_labels->get_num_labels());
      }
      m_kernel->init(data, data);
	}

	initlpsolver();

	weightshistory.clear();

	int32_t numkernels=
         std::dynamic_pointer_cast<CombinedKernel>(m_kernel)->get_num_subkernels();

	::std::vector<float64_t> curweights(numkernels,1.0/numkernels);
	weightshistory.push_back(curweights);

	addingweightsstep(curweights);

	oldalphaterm=curalphaterm;
	oldnormweightssquared=normweightssquared;

	int32_t numberofsilpiterations=0;
	bool final=false;

	while (!final)
	{
		COMPUTATION_CONTROLLERS
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
		auto osvm=svm->get_svm(i);
		auto nsvm=std::make_shared<SVM>(osvm->get_num_support_vectors());

		for (int32_t k=0; k<osvm->get_num_support_vectors() ; k++)
		{
			nsvm->set_alpha(k, osvm->get_alpha(k) );
			nsvm->set_support_vector(k,osvm->get_support_vector(k) );
		}
		nsvm->set_bias(osvm->get_bias() );
		set_svm(i, nsvm);


		osvm=NULL;
	}


	svm=NULL;
	lpw.reset();
	return true;
}




float64_t* MKLMulticlass::getsubkernelweights(int32_t & numweights)
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

void MKLMulticlass::set_mkl_epsilon(float64_t eps )
{
	mkl_eps=eps;
}

void MKLMulticlass::set_max_num_mkliters(int32_t maxnum)
{
	max_num_mkl_iters=maxnum;
}

void MKLMulticlass::set_mkl_norm(float64_t norm)
{
	pnorm=norm;
	if(pnorm<1 )
      error("MKLMulticlass::set_mkl_norm(float64_t norm) : parameter pnorm<1");
}
