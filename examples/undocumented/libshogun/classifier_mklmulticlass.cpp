/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Alexander Binder
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#include <iostream>
#include <shogun/io/SGIO.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/classifier/mkl/MKLMulticlass.h>

// g++ -Wall -O3 classifier_mklmulticlass.cpp -I /home/theseus/private/alx/shoguntrunk/compiledtmp/include -L/home/theseus/private/alx/shoguntrunk/compiledtmp/lib -lshogun

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void print_warning(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void print_error(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

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


void gendata(std::vector<float64_t> & x,std::vector<float64_t> & y,
		CMulticlassLabels*& lab)
{
	int32_t totalsize=240;
	int32_t class1size=80;
	int32_t class2size=70;

	//generating three class data set
	x.resize(totalsize);
	y.resize(totalsize);
	for(size_t i=0; i< x.size();++i)
		getgauss(x[i], y[i]);

	for(size_t i=0; i< x.size();++i)
	{
		if((int32_t)i < class1size)
		{
			x[i]+=0;
			y[i]+=0;
		}
		else if( (int32_t)i< class1size+class2size)
		{
			x[i]+=+1;
			y[i]+=-1;
		}
		else
		{
			x[i]+=-1;
			y[i]+=+1;
		}
	}

	//set labels
	lab=new CMulticlassLabels(x.size());
	for(size_t i=0; i< x.size();++i)
	{
		if((int32_t)i < class1size)
			lab->set_int_label(i,0);
		else if( (int32_t)i< class1size+class2size)
			lab->set_int_label(i,1);
		else
			lab->set_int_label(i,2);
	}
}


void gentrainkernel(float64_t * & ker1 ,float64_t * & ker2, float64_t * & ker3  ,float64_t &
		autosigma,float64_t & n1,float64_t & n2, float64_t & n3,
		const std::vector<float64_t> & x,
		const std::vector<float64_t> & y)
{
	autosigma=0;

	for(size_t l=0; l< x.size();++l)
	{
		for(size_t r=0; r<= l;++r)
		{
			float64_t dist=((x[l]-x[r])*(x[l]-x[r]) + (y[l]-y[r])*(y[l]-y[r]));
			autosigma+=dist*2.0/(float64_t)x.size()/((float64_t)x.size()+1);
		}
	}

	float64_t fm1=0, mean1=0,fm2=0, mean2=0,fm3=0, mean3=0;

	ker1=SG_MALLOC(float64_t,  x.size()*x.size());
	ker2=SG_MALLOC(float64_t,  x.size()*x.size());
	ker3=SG_MALLOC(float64_t,  x.size()*x.size());


	for(size_t l=0; l< x.size();++l)
	{
		for(size_t r=0; r< x.size();++r)
		{
			float64_t dist=((x[l]-x[r])*(x[l]-x[r]) + (y[l]-y[r])*(y[l]-y[r]));

			ker1[l +r*x.size()]=   exp( -dist/autosigma/autosigma) ;
			//ker2[l +r*x.size()]=   exp( -dist/sigma2/sigma2) ;
			ker2[l +r*x.size()]= x[l]*x[r] + y[l]*y[r];

			ker3[l +r*x.size()]= (x[l]*x[r] + y[l]*y[r]+1)*(x[l]*x[r] + y[l]*y[r]+1);

			fm1+=ker1[l +r*x.size()]/(float64_t)x.size()/((float64_t)x.size());
			fm2+=ker2[l +r*x.size()]/(float64_t)x.size()/((float64_t)x.size());
			fm3+=ker3[l +r*x.size()]/(float64_t)x.size()/((float64_t)x.size());

			if(l==r)
			{
				mean1+=ker1[l +r*x.size()]/(float64_t)x.size();
				mean2+=ker2[l +r*x.size()]/(float64_t)x.size();
				mean3+=ker3[l +r*x.size()]/(float64_t)x.size();
			}
		}
	}

	n1=(mean1-fm1);
	n2=(mean2-fm2);
	n3=(mean3-fm3);

	for(size_t l=0; l< x.size();++l)
	{
		for(size_t r=0; r< x.size();++r)
		{
			ker1[l +r*x.size()]=ker1[l +r*x.size()]/n1;
			ker2[l +r*x.size()]=ker2[l +r*x.size()]/n2;
			ker3[l +r*x.size()]=ker3[l +r*x.size()]/n3;
		}
	}
}

void gentestkernel(float64_t * & ker1 ,float64_t * & ker2,float64_t * & ker3,
		const float64_t autosigma,const float64_t n1,const float64_t n2, const float64_t n3,
		const std::vector<float64_t> & x,const std::vector<float64_t> & y,
		const std::vector<float64_t> & tx,const std::vector<float64_t> & ty)
{
	ker1=SG_MALLOC(float64_t,  x.size()*tx.size());
	ker2=SG_MALLOC(float64_t,  x.size()*tx.size());
	ker3=SG_MALLOC(float64_t,  x.size()*tx.size());

	for(size_t l=0; l< x.size();++l)
	{
		for(size_t r=0; r< tx.size();++r)
		{
			float64_t dist=((x[l]-tx[r])*(x[l]-tx[r]) + (y[l]-ty[r])*(y[l]-ty[r]));

			ker1[l +r*x.size()]=   exp( -dist/autosigma/autosigma) ;
			ker2[l +r*x.size()]= x[l]*tx[r] + y[l]*ty[r];
			ker3[l +r*x.size()]= (x[l]*tx[r] + y[l]*ty[r]+1)*(x[l]*tx[r] + y[l]*ty[r]+1);
		}
	}

	for(size_t l=0; l< x.size();++l)
	{
		for(size_t r=0; r< tx.size();++r)
		{
			ker1[l +r*x.size()]=ker1[l +r*x.size()]/n1;
			ker2[l +r*x.size()]=ker2[l +r*x.size()]/n2;
			ker3[l +r*x.size()]=ker3[l +r*x.size()]/n2;

		}
	}
}

void tester()
{
	CMulticlassLabels* lab=NULL;
	std::vector<float64_t> x,y;

	gendata(x,y, lab);
	SG_REF(lab);

	float64_t* ker1=NULL;
	float64_t* ker2=NULL;
	float64_t* ker3=NULL;
	float64_t autosigma=1;
	float64_t n1=0;
	float64_t n2=0;
	float64_t n3=0;

	int32_t numdata=0;
	gentrainkernel( ker1 , ker2, ker3 , autosigma, n1, n2, n3,x,y);
	numdata=x.size();

	CCombinedKernel* ker=new CCombinedKernel();

	CCustomKernel* kernel1=new CCustomKernel();
	CCustomKernel* kernel2=new CCustomKernel();
	CCustomKernel* kernel3=new CCustomKernel();

	kernel1->set_full_kernel_matrix_from_full(SGMatrix<float64_t>(ker1, numdata,numdata,false));
	kernel2->set_full_kernel_matrix_from_full(SGMatrix<float64_t>(ker2, numdata,numdata,false));
	kernel3->set_full_kernel_matrix_from_full(SGMatrix<float64_t>(ker3, numdata,numdata,false));

	SG_FREE(ker1);
	SG_FREE(ker2);
	SG_FREE(ker3);

	ker->append_kernel(kernel1);
	ker->append_kernel(kernel2);
	ker->append_kernel(kernel3);

	//here comes the core stuff
	float64_t regconst=1.0;

	CMKLMulticlass* tsvm =new CMKLMulticlass(regconst, ker, lab);

	tsvm->set_epsilon(0.0001); // SVM epsilon
	// MKL parameters
	tsvm->set_mkl_epsilon(0.01); // subkernel weight L2 norm termination criterion
	tsvm->set_max_num_mkliters(120); // well it will be just three iterations
	tsvm->set_mkl_norm(1.5); // mkl norm
	//starting svm training
	tsvm->train();

	SG_SPRINT("finished svm training\n");

	//starting svm testing on training data
	CMulticlassLabels* res=CLabelsFactory::to_multiclass(tsvm->apply());
	ASSERT(res);

	float64_t err=0;
	for(int32_t i=0; i<numdata;++i)
	{
		ASSERT(i< res->get_num_labels());
		if (lab->get_int_label(i)!=res->get_int_label(i))
			err+=1;
	}
	err/=(float64_t)res->get_num_labels();
	SG_SPRINT("prediction error on training data (3 classes): %f ",err);
	SG_SPRINT("random guess error would be: %f \n",2/3.0);

	//generate test data
	CMulticlassLabels* tlab=NULL;

	std::vector<float64_t> tx,ty;

	gendata( tx,ty,tlab);
	SG_REF(tlab);

	float64_t* tker1=NULL;
	float64_t* tker2=NULL;
	float64_t* tker3=NULL;

	gentestkernel(tker1,tker2,tker3, autosigma, n1,n2,n3, x,y, tx,ty);
	int32_t numdatatest=tx.size();

	CCombinedKernel* tker=new CCombinedKernel();
	SG_REF(tker);
	CCustomKernel* tkernel1=new CCustomKernel();
	CCustomKernel* tkernel2=new CCustomKernel();
	CCustomKernel* tkernel3=new CCustomKernel();

	tkernel1->set_full_kernel_matrix_from_full(SGMatrix<float64_t>(tker1,numdata, numdatatest, false));
	tkernel2->set_full_kernel_matrix_from_full(SGMatrix<float64_t>(tker2,numdata, numdatatest, false));
	tkernel3->set_full_kernel_matrix_from_full(SGMatrix<float64_t>(tker2,numdata, numdatatest, false));

	SG_FREE(tker1);
	SG_FREE(tker2);
	SG_FREE(tker3);

	tker->append_kernel(tkernel1);
	tker->append_kernel(tkernel2);
	tker->append_kernel(tkernel3);

	int32_t numweights;
	float64_t* weights=tsvm->getsubkernelweights(numweights);

	SG_SPRINT("test kernel weights\n");

	for(int32_t i=0; i< numweights;++i)
		SG_SPRINT("%f ", weights[i]);

	SG_SPRINT("\n");

	//set kernel
	tker->set_subkernel_weights(SGVector<float64_t>(weights, numweights));
	tsvm->set_kernel(tker);

	//compute classification error, check mem
	CMulticlassLabels* tres=CLabelsFactory::to_multiclass(tsvm->apply());

	float64_t terr=0;
	for(int32_t i=0; i<numdatatest;++i)
	{
		ASSERT(i< tres->get_num_labels());
		if(tlab->get_int_label(i)!=tres->get_int_label(i))
			terr+=1;
	}
	terr/=(float64_t) tres->get_num_labels();
	SG_SPRINT("prediction error on test data (3 classes): %f ",terr);
	SG_SPRINT("random guess error would be: %f \n",2/3.0);

	SG_UNREF(tsvm);
	SG_UNREF(res);
	SG_UNREF(tres);
	SG_UNREF(lab);
	SG_UNREF(tlab);
	SG_UNREF(tker);

	SG_SPRINT( "finished \n");
}

namespace shogun
{
	extern Version* sg_version;
	extern SGIO* sg_io;
}

int main()
{
	init_shogun(&print_message, &print_warning,
			&print_error);
	try
	{
		sg_version->print_version();
		sg_io->set_loglevel(MSG_INFO);
		tester();
	}
	catch(ShogunException & sh)
	{
		printf("%s",sh.get_exception_string());
	}

	exit_shogun();

	return 0;
}
