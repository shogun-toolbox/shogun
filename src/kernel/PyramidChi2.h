#ifndef PYRAMIDCHI2_H_
#define PYRAMIDCHI2_H_

#include "lib/common.h"
#include "kernel/SimpleKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"


//pyramid classifier over Chi2 matched histograms
//TODO: port to CCombinedKernel (if it is the appropriate) as the pyramid is a weighted linear combination of kernels

class CPyramidChi2 : public CSimpleKernel<DREAL>
{
public:

	CPyramidChi2(INT size, DREAL width2,
		INT* pyramidlevels2,INT numlevels2,
		INT  numbinsinhistogram2, DREAL* weights2,INT numweights2);

	virtual bool init(CFeatures* l, CFeatures* r);

	CPyramidChi2(CRealFeatures* l, CRealFeatures* r, INT size, DREAL width2,
		INT* pyramidlevels2,INT numlevels2,
		INT  numbinsinhistogram2, DREAL* weights2,INT numweights2);

	virtual ~CPyramidChi2();

	virtual void cleanup();

	/// load and save kernel init_data
	virtual bool load_init(FILE* src);
	virtual bool save_init(FILE* dest);

	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type()
	{
		//preliminary output
		return K_PYRAMIDCHI2;
	}

	// return the name of a kernel
	virtual const CHAR* get_name()
	{
		return("PyramidoverChi2\0");
	}

	void setstandardweights(); // sets weights
	bool sanitycheck_weak(); // performs a weak check, does not test for correct feature length

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	virtual DREAL compute(INT idx_a, INT idx_b);

protected:
	DREAL width;
	INT* pyramidlevels;
	INT numlevels; // length of vector pyramidlevels
	INT numbinsinhistogram;
	DREAL* weights;
	INT numweights; // length of vector weights
	//bool sanitycheckbit;
};

#endif /*PYRAMIDCHI2_H_*/
