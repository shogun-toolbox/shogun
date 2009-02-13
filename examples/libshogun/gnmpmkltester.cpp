/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Alexander Binder
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */



// *******************************************************************
// application example for class CGMNPMKL
//
// compile it with
// g++ -I. -I<your glpk installation path>/include -I<your glpk shogun installation path>/include/shogun -O2 -g3 -Wall gnmpmkltester.cpp -L<your glpk shogun installation path>/lib -lshogun -o"shoguncgmnpmkltester"  
//
// ************************************************************************




#include <iostream>

//for the testing method
#include <shogun/classifier/svm/gmnpmkl.h>

#include <shogun/lib/io.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/CombinedKernel.h> 
#include <shogun/features/DummyFeatures.h>

void getgauss(float64_t & y1, float64_t & y2)
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


void gendata(std::vector<float64_t> & x,std::vector<float64_t> & y, CLabels*& lab)
{
	//SG_SPRINT("generating three class data\n");
	
	x.resize(720);
	y.resize(720);
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
	

	//set labels
	lab=new CLabels(x.size());
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
}


void gentrainkernel(float64_t * & ker1 ,float64_t * & ker2 ,float64_t & autosigma,float64_t & n1,float64_t & n2, const std::vector<float64_t> & x,const std::vector<float64_t> & y)
{

	autosigma=0;
	

	
	//std::vector< std::vector<  float64_t >  > ker1( x.size(), std::vector<float64_t>(x.size()) ),ker2( x.size(), std::vector<float64_t>(x.size()) );
	for(size_t l=0; l< x.size();++l)
	{
		for(size_t r=0; r<= l;++r)
		{
			float64_t dist=((x[l]-x[r])*(x[l]-x[r]) + (y[l]-y[r])*(y[l]-y[r]));
			autosigma+=dist*2.0/(float64_t)x.size()/((float64_t)x.size()+1);
		}
	}	

	
	
	//SG_SPRINT("estimated kernel width %f \n", autosigma);
	
	float64_t fm1=0, mean1=0,fm2=0, mean2=0;
	
	
	
	ker1=new float64_t[ x.size()*x.size()];
	ker2=new float64_t[ x.size()*x.size()];

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

n1=(mean1-fm1);
n2=(mean2-fm2);
	//SG_SPRINT("estimated kernel normalizations %f %f \n", (mean1-fm1),(mean2-fm2));


	for(size_t l=0; l< x.size();++l)
	{
		for(size_t r=0; r< x.size();++r)
		{
			ker1[l +r*x.size()]=ker1[l +r*x.size()]/n1;
			ker2[l +r*x.size()]=ker2[l +r*x.size()]/n2;

		}
	}
}

void gentestkernel(float64_t * & ker1 ,float64_t * & ker2 ,const float64_t  autosigma,const float64_t n1,const float64_t  n2, const std::vector<float64_t> & x,const std::vector<float64_t> & y,const std::vector<float64_t> & tx,const std::vector<float64_t> & ty)
{
		ker1=new float64_t[ x.size()*tx.size()];
	ker2=new float64_t[ x.size()*tx.size()];

	for(size_t l=0; l< x.size();++l)
	{
		for(size_t r=0; r< tx.size();++r)
		{

			
			float64_t dist=((x[l]-tx[r])*(x[l]-tx[r]) + (y[l]-ty[r])*(y[l]-ty[r]));
			
			ker1[l +r*x.size()]=   exp( -dist/autosigma/autosigma) ;
			//ker2[l +r*x.size()]=   exp( -dist/sigma2/sigma2) ;
			ker2[l +r*x.size()]= x[l]*tx[r] + y[l]*ty[r];

		}
	}


	for(size_t l=0; l< x.size();++l)
	{
		for(size_t r=0; r< tx.size();++r)
		{
			ker1[l +r*x.size()]=ker1[l +r*x.size()]/n1;
			ker2[l +r*x.size()]=ker2[l +r*x.size()]/n2;

		}
	}
}

void tester()
{
	init_shogun();

	
	



CLabels* lab(NULL); 
std::vector<float64_t> x,y;

gendata( x,y,lab);

float64_t * ker1(NULL);
float64_t * ker2(NULL);
float64_t autosigma=1;
float64_t n1=0;
float64_t n2=0;	
int32_t numdata=0;
gentrainkernel( ker1 , ker2 , autosigma, n1, n2,x,y);
numdata=x.size();

	CCombinedFeatures *l(NULL), *r(NULL);
	
	l=new  CCombinedFeatures;
	r=new  CCombinedFeatures;
	
	l->append_feature_obj(new CDummyFeatures(numdata));
	l->append_feature_obj(new CDummyFeatures(numdata));
	
	r->append_feature_obj(new CDummyFeatures(numdata));
	r->append_feature_obj(new CDummyFeatures(numdata));
	
		printf("here0, %d\n",numdata);

	CCombinedKernel * ker=new CCombinedKernel(l,r);
	
	printf("here1\n");


	CCustomKernel *kernel1(NULL),*kernel2(NULL);
	
	printf("here\n");


	kernel1=new CCustomKernel;
	kernel2=new CCustomKernel;
	kernel1->set_full_kernel_matrix_from_full(ker1,numdata, numdata);
	kernel2->set_full_kernel_matrix_from_full(ker2,numdata, numdata);
	
	ker->append_kernel(kernel1);  	
	ker->append_kernel(kernel2);  
	

	
	
	//here comes the core stuff
	float64_t regconst=1.0;

	CGMNPMKL * tsvm =new CGMNPMKL(regconst, ker, lab);
	
	tsvm->set_epsilon(0.0001); // SVM epsilon
	// MKL parameters
	tsvm->thresh=0.01; // subkernel weight L2 norm termination criterion
	tsvm->maxiters=120; // well it will be just three iterations
	
	//SG_SPRINT("starting svm training\n");
	tsvm->train();
	SG_SPRINT("finished svm training\n");
	
	CLabels *res(NULL), *quirk(NULL);
	
	//SG_SPRINT("starting svm testing on training data\n");
	res=tsvm->classify(quirk);
	
	float64_t err=0;
	for(int32_t i=0; i<numdata;++i)
	{
		
		ASSERT(i< res->get_num_labels());
		//SG_SPRINT("at index i= %d truelabel= %d predicted= %d \n",i,lab->get_int_label(i),res->get_int_label(i));
		if(lab->get_int_label(i)!=res->get_int_label(i))
		{
			err+=1;
		}
	}
	err/=(float64_t)res->get_num_labels();
	//SG_SPRINT("prediction error on training data (3 classes): %f",err);
	//SG_SPRINT("random guess error would be: %f \n",2/3.0);
	

	delete[] ker1;
	ker1=NULL;
	delete[] ker2;
	ker2=NULL;

	SG_UNREF(ker);
	ker=NULL;
	
	SG_UNREF(l);
	l=NULL;
	SG_UNREF(r);
	r=NULL;
	SG_UNREF(lab);
	
	SG_UNREF(res);
	res=NULL;
	SG_UNREF(quirk);
	quirk=NULL;

	SG_UNREF(kernel1);
	kernel1=NULL;
	SG_UNREF(kernel2);
	kernel2=NULL;

	
	//SG_SPRINT("generating test data\n");
	

	



CLabels* tlab(NULL); 

std::vector<float64_t> tx,ty;

gendata( tx,ty,tlab);


float64_t * tker1(NULL);
float64_t * tker2(NULL);

gentestkernel( tker1 , tker2 , autosigma, n1, n2,x,y,tx,ty);
int32_t numdatatest=tx.size();
	
	CCombinedFeatures *tl(NULL), *tr(NULL);
	
	tl=new  CCombinedFeatures;
	tr=new  CCombinedFeatures;
	
	tl->append_feature_obj(new CDummyFeatures(numdatatest));
	tl->append_feature_obj(new CDummyFeatures(numdatatest));
	
	tr->append_feature_obj(new CDummyFeatures(numdatatest));
	tr->append_feature_obj(new CDummyFeatures(numdatatest));
	
	
	CCombinedKernel * tker=new CCombinedKernel(tl,tr);
	
	CCustomKernel *tkernel1(NULL),*tkernel2(NULL);
	
	tkernel1=new CCustomKernel;
	tkernel2=new CCustomKernel;
	
	tkernel1->set_full_kernel_matrix_from_full(tker1,numdata, numdatatest);
	tkernel2->set_full_kernel_matrix_from_full(tker2,numdata, numdatatest);

	tker->append_kernel(tkernel1);  	
	tker->append_kernel(tkernel2);  
	
	int32_t numweights;
	float64_t* weights=tsvm-> getsubkernelweights(numweights);
	
	//SG_SPRINT("test kernel weights (I always forget to set them)\n");
	for(int32_t i=0; i< numweights;++i)
	{
		SG_SPRINT("%f ", weights[i]);
	}
	//SG_SPRINT("\n");
	//set kernel
	tker->set_subkernel_weights(weights, numweights);
	tsvm->set_kernel(tker);
	
	
	//TODO: compute classif error, check mem
	CLabels *tres(NULL), *tquirk(NULL);
	tres=tsvm->classify(tquirk);
	
	float64_t terr=0;
	for(int32_t i=0; i<numdatatest;++i)
	{
		
		ASSERT(i< tres->get_num_labels());
		//SG_SPRINT("at index i= %d truelabel= %d predicted= %d \n",i,lab->get_int_label(i),tres->get_int_label(i));
		if(tlab->get_int_label(i)!=tres->get_int_label(i))
		{
			terr+=1;
		}
	}
	terr/=(float64_t)tres->get_num_labels();
	SG_SPRINT("prediction error on test data (3 classes): %f",terr);
	SG_SPRINT("random guess error would be: %f \n",2/3.0);
	

	
	
	delete[] tker1;
	tker1=NULL;
	delete[] tker2;
	tker2=NULL;

	SG_UNREF(tker);
	tker=NULL;
	
	SG_UNREF(tl);
	tl=NULL;
	SG_UNREF(tr);
	tr=NULL;
	SG_UNREF(tlab);
	
	SG_UNREF(tres);
	tres=NULL;
	SG_UNREF(tquirk);
	tquirk=NULL;

	SG_UNREF(tkernel1);
	tkernel1=NULL;
	SG_UNREF(tkernel2);
	tkernel2=NULL;

	delete[] weights;
	weights=NULL;
	
	//SG_SPRINT( "finished \n");
}


int main()
{
	
	
	try{
		tester();
	}
	catch(ShogunException & sh)
	{
		printf("%s",sh.get_exception_string());
	}
	
	//SG_SPRINT("finished \n");
	
	
}