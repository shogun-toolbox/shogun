#include "gmnpmkl.h"

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
	
	assert((int)normw2.size()==numkernels);
	assert(sumofpositivealphas>=0);

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
	//assert(true==addedconstraint);

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

	// to avoid leaks
#if !defined(HAVE_SWIG) || defined(HAVE_R)
	if(svm!=NULL)
	{
		CLabels* lb=svm->get_labels();
		delete lb;
		lb=NULL;
	}
#endif	
	
	delete svm;
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
	
	if(svm!=NULL)
	{
		// to avoid leaks
	#if !defined(HAVE_SWIG) || defined(HAVE_R)
		if(svm!=NULL)
		{
			CLabels* lb=svm->get_labels();
			delete lb;
			lb=NULL;
		}
	#endif	
		
		delete svm;
	}
	svm=new CGMNPSVM;
	
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

	//is a leak if SWIG if false ==  #if defined(HAVE_SWIG) && !defined(HAVE_R)
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
		
		assert(wold.size()==wnew.size());
		
		for(size_t i=0;i< wnew.size();++i)
		{
			delta+=(wold[i]-wnew[i])*(wold[i]-wnew[i]);
		}
		delta=sqrt(delta);

		SG_PRINT( "CGMNPMKL::evaluatefinishcriterion(): L2 norm of changes= %f, required for termination by member variables thresh= %f \n",delta, thresh);

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
	float64_t * lab=labels->get_labels ( tmpint);
	std::copy(lab,lab+labels->get_num_labels(), trainlabels2.begin());
	delete[] lab;
	lab=NULL;
	
	
	assert(trainlabels2.size()>0);
	float64_t sum=0;

	for(int32_t nc=0; nc< labels->get_num_classes();++nc)
	{
		float64_t bia=svm->get_svm(nc)->get_bias();
		sum+= bia*bia;
	}
	
	::std::vector< ::std::vector<float64_t> > basealphas;
	svm->getbasealphas( basealphas);

	for(size_t lb=0; lb< trainlabels2.size();++lb)
	{
		for(int32_t nc=0; nc< labels->get_num_classes();++nc)
		{
			if((int)nc!=trainlabels2[lb])
			{
				
				float64_t bia1=svm->get_svm(trainlabels2[lb])->get_bias();
				float64_t bia2=svm->get_svm(nc)->get_bias();
				
				sum+= -basealphas[nc][lb]*(bia1-bia2-1);
			}

		}
	}

	return(sum);
}

float64_t CGMNPMKL::getsquarenormofprimalcoefficients(
		const int32_t ind)
{
	// alphas are already correctly transformed!

	float64_t tmp=0;

	for(int32_t classindex=0; classindex< labels->get_num_classes();++classindex)
	{

		for (int32_t i=0; i < svm->get_svm(classindex)->get_num_support_vectors(); ++i)
		{
			float64_t alphai=svm->get_svm(classindex)->get_alpha(i);// svmstuff[classindex].alphas[i];
			int32_t svindi= svm->get_svm(classindex)->get_support_vector(i); //svmstuff[classindex].svind[i];

			for (int32_t k=0; k < svm->get_svm(classindex)->get_num_support_vectors(); ++k)
			{
				float64_t alphak=svm->get_svm(classindex)->get_alpha(k);
				int32_t svindk=svm->get_svm(classindex)->get_support_vector(k);

				tmp+=alphai*dynamic_cast<CCombinedKernel *>(kernel)->get_kernel(ind)->kernel(svindi,svindk)
				*alphak;

			}
		}
	}
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

			SG_PRINT("SILP iteration %d weights in silp \n",numberofsilpiterations);
			for(size_t i=0; i< curweights.size();++i)
			{
				SG_PRINT("%f ",curweights[i]);
			}
			SG_PRINT("\n");			

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

			osvm=NULL;
			set_svm(i, nsvm);
		}
		delete svm;
		svm=NULL;
		return(true);
}

void CGMNPMKL::tester()
{
	// generates three class problem, 210, 240, 270 data points for the classes
	
	SG_PRINT("generating data\n");
	
	std::vector<float64_t> x(720);
	std::vector<float64_t> y(720);
	
	for(size_t i=0; i< x.size();++i)
	{	
		getgauss(x[i], y[i]);
	}
	
	for(size_t i=0; i< x.size();++i)
	{	
		if(i < 210)
		{
			x[i]+= 0;
			y[i]+=	0;
		} 
		else if( i< 450)
		{
			x[i]+= 1;
			y[i]+=	-1;
		}
		else
		{
			x[i]+= -1;
			y[i]+=	+1;
		}
	}
	
//	float64_t sigma1=0.5, sigma2=1;
	
	float64_t autosigma=0;
	
	float64_t * ker1(NULL),*ker2 (NULL);
	
	ker1=new float64_t[ x.size()*x.size()];
	ker2=new float64_t[ x.size()*x.size()];
	
	//std::vector< std::vector<  float64_t >  > ker1( x.size(), std::vector<float64_t>(x.size()) ),ker2( x.size(), std::vector<float64_t>(x.size()) );
	for(size_t l=0; l< x.size();++l)
	{
		for(size_t r=0; r<= l;++r)
		{
			float64_t dist=((x[l]-x[r])*(x[l]-x[r]) + (y[l]-y[r])*(y[l]-y[r]));
			autosigma+=dist*2.0/(float64_t)x.size()/((float64_t)x.size()+1);
		}
	}	
	
	SG_PRINT("estimated kernel width %f \n", autosigma);
	
	float64_t fm1=0, mean1=0,fm2=0, mean2=0;
	
	for(size_t l=0; l< x.size();++l)
	{
		for(size_t r=0; r< x.size();++r)
		{

			
			float64_t dist=((x[l]-x[r])*(x[l]-x[r]) + (y[l]-y[r])*(y[l]-y[r]));
			
			ker1[l +r*x.size()]=   exp( -dist/autosigma/autosigma) ;
			//ker2[l +r*x.size()]=   exp( -dist/sigma2/sigma2) ;
			ker2[l +r*x.size()]= x[l]*x[r] + y[l]*y[r];
			
			fm1+=ker1[l +r*x.size()]/(float64_t)x.size()/((float64_t)x.size());
			fm2+=ker2[l +r*x.size()]/(float64_t)x.size()/((float64_t)x.size());
			
			if(l==r)
			{
				mean1+=ker1[l +r*x.size()]/(float64_t)x.size();
				mean2+=ker2[l +r*x.size()]/(float64_t)x.size();
			}
		}
	}
	SG_PRINT("estimated kernel normalizations %f %f \n", (mean1-fm1),(mean2-fm2));
	
	for(size_t l=0; l< x.size();++l)
	{
		for(size_t r=0; r< x.size();++r)
		{
			ker1[l +r*x.size()]=ker1[l +r*x.size()]/(mean1-fm1);
			ker2[l +r*x.size()]=ker2[l +r*x.size()]/(mean2-fm2);

		}
	}
	
	
	CCombinedFeatures *l(NULL), *r(NULL);
	
	l=new  CCombinedFeatures;
	r=new  CCombinedFeatures;
	
	l->append_feature_obj(new CDummyFeatures(720));
	l->append_feature_obj(new CDummyFeatures(720));
	
	r->append_feature_obj(new CDummyFeatures(720));
	r->append_feature_obj(new CDummyFeatures(720));
	
	
	CCombinedKernel * ker=new CCombinedKernel(l,r);
	
	CCustomKernel *kernel1(NULL),*kernel2(NULL);
	
	kernel1=new CCustomKernel;
	kernel2=new CCustomKernel;
	
	kernel1->set_full_kernel_matrix_from_full(ker1,x.size(), x.size());
	kernel2->set_full_kernel_matrix_from_full(ker2,x.size(), x.size());
	
	//printf( "k1 %f %f  ",ker1[721],  kernel1->kernel(1,1) );
	//printf( "k2 %f %f ", ker2[721], kernel2->kernel(1,1) );
	
	ker->append_kernel(kernel1);  	
	ker->append_kernel(kernel2);  
	
	
	//set labels
	CLabels* lab=new CLabels(x.size());
	for(size_t i=0; i< x.size();++i)
	{	
		if(i < 210)
		{
			lab->set_int_label(i,0);
		} 
		else if( i< 450)
		{
			lab->set_int_label(i,1);
		}
		else
		{
			lab->set_int_label(i,2);
		}
	}
	
	float64_t regconst=1.0;

	CGMNPMKL * tsvm =new CGMNPMKL(regconst, ker, lab);
	
	tsvm->set_epsilon(0.0001);
	SG_PRINT("starting svm training\n");
	
	tsvm->train();
	
	SG_PRINT("finished svm training\n");
	
	//TODO: compute classif error, check mem
	CLabels *res(NULL), *quirk(NULL);
	
	SG_PRINT("starting svm testing on training data\n");
	res=tsvm->classify(quirk);
	
	float64_t err=0;
	for(int32_t i=0; i<720;++i)
	{
		
		ASSERT(i< res->get_num_labels());
		//printf("at index i= %d truelabel= %d predicted= %d \n",i,lab->get_int_label(i),res->get_int_label(i));
		if(lab->get_int_label(i)!=res->get_int_label(i))
		{
			err+=1;
		}
	}
	err/=(float64_t)res->get_num_labels();
	printf("prediction error on training data (3 classes): %f",err);
	printf("random guess error would be: %f \n",2/3.0);
	
#if !defined(HAVE_SWIG) || defined(HAVE_R)
	delete ker;
	ker=NULL;
#endif	
	
	delete[] ker1;
	ker1=NULL;
	delete[] ker2;
	ker2=NULL;
	

	delete res;
	res=NULL;
	delete quirk;
	quirk=NULL;
	
	delete l;
	l=NULL;
	delete r;
	r=NULL;
	/*
	delete kernel1;
	kernel1=NULL;
	delete kernel2;
	kernel2=NULL;
	*/
	
	
	SG_PRINT("generating test data\n");
	
	std::vector<float64_t> tx(720);
	std::vector<float64_t> ty(720);
	
	for(size_t i=0; i< tx.size();++i)
	{	
		getgauss(tx[i], ty[i]);
	}
	
	for(size_t i=0; i< tx.size();++i)
	{	
		if(i < 210)
		{
			tx[i]+= 0;
			ty[i]+=	0;
		} 
		else if( i< 450)
		{
			tx[i]+= 1;
			ty[i]+=	-1;
		}
		else
		{
			tx[i]+= -1;
			ty[i]+=	+1;
		}
	}
	
	
	float64_t * tker1(NULL),*tker2 (NULL);
	
	tker1=new float64_t[ x.size()*tx.size()];
	tker2=new float64_t[ x.size()*tx.size()];
	
	//std::vector< std::vector<  float64_t >  > ker1( x.size(), std::vector<float64_t>(x.size()) ),ker2( x.size(), std::vector<float64_t>(x.size()) );
	for(size_t l2=0; l2< x.size();++l2)
	{
		for(size_t r2=0; r2< tx.size();++r2)
		{

			
			float64_t dist=(x[l2]-tx[r2])*(x[l2]-tx[r2]) + (y[l2]-ty[r2])*(y[l2]-ty[r2]);
			tker1[l2 +r2*x.size()]=   exp( -dist/autosigma/autosigma) ;
			//tker2[l2 +r2*x.size()]=   exp( -dist/sigma2/sigma2) ;
			tker2[l2 +r2*x.size()]= x[l2]*tx[r2] + y[l2]*ty[r2];
		}
	}
	
	
	for(size_t l2=0; l2< x.size();++l2)
	{
		for(size_t r2=0; r2< tx.size();++r2)
		{
			tker1[l2 +r2*x.size()]=tker1[l2 +r2*x.size()]/(mean1-fm1);
			tker2[l2 +r2*x.size()]=tker2[l2 +r2*x.size()]/(mean2-fm2);

		}
	}
	
	
	CCombinedFeatures *tl(NULL), *tr(NULL);
	
	tl=new  CCombinedFeatures;
	tr=new  CCombinedFeatures;
	
	tl->append_feature_obj(new CDummyFeatures(720));
	tl->append_feature_obj(new CDummyFeatures(720));
	
	tr->append_feature_obj(new CDummyFeatures(720));
	tr->append_feature_obj(new CDummyFeatures(720));
	
	
	CCombinedKernel * tker=new CCombinedKernel(tl,tr);
	
	CCustomKernel *tkernel1(NULL),*tkernel2(NULL);
	
	tkernel1=new CCustomKernel;
	tkernel2=new CCustomKernel;
	
	tkernel1->set_full_kernel_matrix_from_full(tker1,x.size(), tx.size());
	tkernel2->set_full_kernel_matrix_from_full(tker2,x.size(), tx.size());

	tker->append_kernel(tkernel1);  	
	tker->append_kernel(tkernel2);  
	
	tsvm->set_kernel(tker);
	
	
	//TODO: compute classif error, check mem
	CLabels *tres(NULL), *tquirk(NULL);
	tres=tsvm->classify(tquirk);
	
	float64_t terr=0;
	for(int32_t i=0; i<720;++i)
	{
		
		ASSERT(i< tres->get_num_labels());
		//printf("at index i= %d truelabel= %d predicted= %d \n",i,lab->get_int_label(i),tres->get_int_label(i));
		if(lab->get_int_label(i)!=tres->get_int_label(i))
		{
			terr+=1;
		}
	}
	terr/=(float64_t)tres->get_num_labels();
	printf("prediction error on test data (3 classes): %f",terr);
	printf("random guess error would be: %f \n",2/3.0);
	

#if !defined(HAVE_SWIG) || defined(HAVE_R)
	delete tker;
	tker=NULL;
#endif	
	
	
	delete[] tker1;
	tker1=NULL;
	delete[] tker2;
	tker2=NULL;
	

	delete tres;
	tres=NULL;
	delete tquirk;
	tquirk=NULL;
	
	
	delete tl;
	tl=NULL;
	delete tr;
	tr=NULL;
	/*
	//delete tkernel1;
	tkernel1=NULL;
	//delete tkernel2;
	tkernel2=NULL;
	*/
	
	
	
	
	
	
	delete tsvm;
	tsvm=NULL;
	delete lab;
	lab=NULL;

	
	printf( "finished \n");
}

void CGMNPMKL::getgauss(float64_t & y1, float64_t & y2)
{
    float x1, x2, w;

    do {
            x1 = 2.0 * rand()/(float64_t)RAND_MAX - 1.0;
            x2 = 2.0 * rand()/(float64_t)RAND_MAX - 1.0;
            w = x1 * x1 + x2 * x2;
    } while ( (w >= 1.0)|| (w<1e-9) );
    
    w = sqrt( (-2.0 * log( w ) ) / w );
    y1 = x1 * w;
    y2 = x2 * w;

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

