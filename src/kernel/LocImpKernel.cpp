#include <assert.h>
#include "lib/common.h"
#include "kernel/LocImpKernel.h"

CLocImpKernel::CLocImpKernel(int width_, int degree1_, 
			     int degree2_) 
  : CKernel(), width(width_), degree1(degree1_), degree2(degree2_), dim(0)
{
  
} ;

CLocImpKernel::~CLocImpKernel() 
{
}
  
void CLocImpKernel::init(CFeatures* f)
{

  PYRAL=0 ; 
  NORM=1.0 ;
  pyra=NULL;
  stage1=NULL;
  NOF_NTS=0 ;

  /* convolution, potentiation */
  double conv;
  double pot;
  double sum;
  /* ... */
  int i,j;
  
  /* pyramid */
  NOF_NTS=dim ;
  PYRAL = 2 * width - 1;  /* size of pyra vector */
  NORM = 1.0 / (width * width);
  
  /* check assumptions */
  assert((degree1 & ~0x7) == 0);
  assert((degree2 & ~0x7) == 0);
  
  /* build pyramid, [but the normalization is deferred] */
  /* [matlab code]: pyra = conv(ones(1,width),ones(1,width)) / (width*width); */
  pyra = (int*) new int[((PYRAL+1) * sizeof(double))];
  for (j = 0; j < PYRAL; ++j) {
    pyra[j] = (j < width) ? j+1 : PYRAL-j;
    /* defer normalization */
    /***  pyra[j] *= NORM; */
  }
  
  /* allocate memory for pointwise dot product */
  stage1 = (int *) new int[NOF_NTS+1];

} ;

void CLocImpKernel::cleanup()
{
  delete[] pyra ;
  delete[] stage1 ;
} ;

REAL CLocImpKernel::compute(CFeatures* a, long idx_a, CFeatures* b, long idx_b)
{
  short int *avec, *bvec ; /* do something here */
  double dpt ;
  
  dot_pyr(&dpt, avec, bvec, 1,1) ;
  
  return dpt ;
  
}


void CLocImpKernel::dot_pyr (double* dpt,
		const short int* const data1,
		const short int* const data2,
		const int num1,
		const int num2)
{
   /* matrix */
   const int symetrical = (data1 == data2);
   const short int* x1;
   const short int* x2;
   /* convolution, potentiation */
   double conv;
   double pot;
   double sum;
   /* ... */
   int m1;
   int m2;
   int i;
   int j;

   /* check assumptions */
   if (symetrical) assert(num1 == num2);
   assert((degree1 & ~0x7) == 0);
   assert((degree2 & ~0x7) == 0);

   /* LOOP */
   x1 = data1;
   for (m1 = 0; m1 < num1; ++m1) {
      /* x1 = &(data1[m1*NOF_NTS]); */
      x2 = data2;
      if (symetrical) x2 += m1*NOF_NTS;
      for (m2 = (symetrical ? m1 : 0); m2 < num2; ++m2) {
	 /* x2 = &(data2[m2*NOF_NTS]); */
	 
	 /* calculate dot product nucleotide-wise */
	 /* [matlab code]: pointwise = x1 .* x2; */
	 /* [matlab code]: pointwise = sum(reshape(pointwise,NOF_BITS,nof_nts));  %represent matching nucleotides */
	 for (i = 0; i < NOF_NTS; ++i) {
	    stage1[i] = (x1[i] == *x2) ? 1 : 0;
	    ++x2;
	 }
	 /* [matlab code]: stage1 = conv(pointwise,pyra);      %has length nof_nts+pyral-1 */
	 /* [matlab code]: stage1 = stage1(pyral:nof_nts);     %only use entries where the pyra did fit into it */
	 /* [matlab code]: stage1 = stage1 .^ degree1;         %first nonlinearity */
	 /* [matlab code]: dpt_ij = sum(stage1) .^ degree2;    %second nonlinearity */
	 sum = 0.0;
	 for (i = 0; i < NOF_NTS-PYRAL+1; ++i) {
	    /* calculate convolution pointwise */
	    conv = 0.0;
	    for (j = 0; j < PYRAL; ++j) {
	       conv += stage1[i+j] * pyra[j]; 
	    }
	    /* now, make up for deferred normalization */
	    conv *= NORM; 
	    /* pot = conv ^ degree1; */
	    pot = ((degree1 & 0x1) == 0) ? 1.0 : conv;
	    if ((degree1 & ~0x1) != 0) {
	       conv *= conv;
	       if ((degree1 & 0x2) != 0) pot *= conv;
	       if ((degree1 & ~0x3) != 0) {
		  conv *= conv;
		  if ((degree1 & 0x4) != 0) pot *= conv;
	       }
	       } 
	    /* don't need to store, just add */
	    sum += pot;
	 }
	 /* pot = sum ^ degree2; */
	 pot = ((degree2 & 0x1) == 0) ? 1.0 : sum;
	 if ((degree2 & ~0x1) != 0) {
	    sum *= sum;
	    if ((degree2 & 0x2) != 0) pot *= sum;
	    if ((degree2 & ~0x3) != 0) {
	       sum *= sum;
	       if ((degree2 & 0x4) != 0) pot *= sum;
	    }
	 }
	 /* save result to target matrix */
	 /*	 if (m1 == 0 && m2 == 0) printf ("0,0: %f\n", pot);
		 if (m1 == 0 && m2 == 1) printf ("0,1: %f\n", pot);*/
	 dpt[m1+num1*m2] = pot;
	 if (symetrical) {
	    dpt[m2+num1*m1] = pot;
	 }
      }
      x1 += NOF_NTS;
   }
}
