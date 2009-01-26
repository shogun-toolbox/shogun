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



#include "classifier/svm/gmnpmkl.h"

#include <iostream>

//for the testing method
#include "kernel/CustomKernel.h" 
#include "kernel/CombinedKernel.h" 
#include "features/DummyFeatures.h"

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

void tester()
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
	
	
	//here comes the core stuff
	float64_t regconst=1.0;

	CGMNPMKL * tsvm =new CGMNPMKL(regconst, ker, lab);
	
	tsvm->set_epsilon(0.0001); // SVM epsilon
	SG_PRINT("starting svm training\n");
	
	
	// MKL parameters
	tsvm->thresh=0.01; // subkernel weight L2 norm termination criterion
	tsvm->maxiters=120; // well it will be just three iterations
	
	tsvm->train();
	
	SG_PRINT("finished svm training\n");
	
	CLabels *res(NULL), *quirk(NULL);
	
	SG_PRINT("starting svm testing on training data\n");
	res=tsvm->classify(quirk);
	
	float64_t err=0;
	for(int32_t i=0; i<720;++i)
	{
		
		ASSERT(i< res->get_num_labels());
		//SG_PRINT("at index i= %d truelabel= %d predicted= %d \n",i,lab->get_int_label(i),res->get_int_label(i));
		if(lab->get_int_label(i)!=res->get_int_label(i))
		{
			err+=1;
		}
	}
	err/=(float64_t)res->get_num_labels();
	SG_PRINT("prediction error on training data (3 classes): %f",err);
	SG_PRINT("random guess error would be: %f \n",2/3.0);
	
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
	
	int32_t numweights;
	float64_t* weights=tsvm-> getsubkernelweights(numweights);
	
	SG_PRINT("test kernel weights (I always forget to set them)\n");
	for(int32_t i=0; i< numweights;++i)
	{
		SG_PRINT("%f ", weights[i]);
	}
	SG_PRINT("\n");
	//set kernel
	tker->set_subkernel_weights(weights, numweights);
	tsvm->set_kernel(tker);
	
	
	//TODO: compute classif error, check mem
	CLabels *tres(NULL), *tquirk(NULL);
	tres=tsvm->classify(tquirk);
	
	float64_t terr=0;
	for(int32_t i=0; i<720;++i)
	{
		
		ASSERT(i< tres->get_num_labels());
		//SG_PRINT("at index i= %d truelabel= %d predicted= %d \n",i,lab->get_int_label(i),tres->get_int_label(i));
		if(lab->get_int_label(i)!=tres->get_int_label(i))
		{
			terr+=1;
		}
	}
	terr/=(float64_t)tres->get_num_labels();
	SG_PRINT("prediction error on test data (3 classes): %f",terr);
	SG_PRINT("random guess error would be: %f \n",2/3.0);
	

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
	
	delete tsvm;
	tsvm=NULL;
	delete lab;
	lab=NULL;

	delete[] weights;
	weights=NULL;
	
	SG_PRINT( "finished \n");
}


int main()
{
	
	
	try{
		tester();
	}
	catch(ShogunException & sh)
	{
		SG_ERROR(sh.get_exception_string());
	}
	
	SG_PRINT("finished \n");
	
	
}