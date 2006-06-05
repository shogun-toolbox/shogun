#include "lib/common.h"
#include "lib/io.h"
#include "kernel/SimpleLocalityImprovedCharKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"

CSimpleLocalityImprovedCharKernel::CSimpleLocalityImprovedCharKernel(LONG size, INT l, INT d1, INT d2)
  : CCharKernel(size),length(l),inner_degree(d1),outer_degree(d2),match(NULL), pyramid_weights(NULL)
{
}

CSimpleLocalityImprovedCharKernel::~CSimpleLocalityImprovedCharKernel() 
{
	cleanup();
}

bool CSimpleLocalityImprovedCharKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CCharKernel::init(l,r,do_init);

	if (result)
	  {
	    INT num_features = ((CCharFeatures*) l)->get_num_features() ;
	    match=new CHAR[num_features];
	    pyramid_weights = new DREAL[num_features] ;
	    CIO::message(M_INFO, "initializing pyramid weights: size=%ld length=%i\n", num_features, length) ;

	    const INT PYRAL = 2 * length - 1; // total window length
	    DREAL PYRAL_pot;
	    INT DEGREE1_1  = (inner_degree & 0x1)==0 ;
	    INT DEGREE1_1n = (inner_degree & ~0x1)!=0 ;
	    INT DEGREE1_2  = (inner_degree & 0x2)!=0 ;
	    INT DEGREE1_3  = (inner_degree & ~0x3)!=0 ;
	    INT DEGREE1_4  = (inner_degree & 0x4)!=0 ;
	    {
	      DREAL PYRAL_ = PYRAL ;
	      PYRAL_pot = DEGREE1_1 ? 1.0 : PYRAL_;
	      if (DEGREE1_1n) {
		PYRAL_ *= PYRAL_;
		if (DEGREE1_2) PYRAL_pot *= PYRAL_;
		if (DEGREE1_3) {
		  PYRAL_ *= PYRAL_ ;
		  if (DEGREE1_4) PYRAL_pot *= PYRAL_;
		}
	      }
	    } 
	    
	    INT pyra_len  = num_features - PYRAL + 1 ;
	    INT pyra_len2 = (int) pyra_len / 2 ;
	    {
	      INT j ;
	      for (j = 0; j < pyra_len; j++) 
		pyramid_weights[j] = 4*((DREAL)((j < pyra_len2) ? j+1 : pyra_len-j))/((DREAL)pyra_len) ;
	      for (j = 0; j < pyra_len; j++) 
		pyramid_weights[j] /= PYRAL_pot ;
	    } 
	    
	  } ;

	return (match!=NULL && result==true);
}
  
void CSimpleLocalityImprovedCharKernel::cleanup()
{
	delete[] match;
	match=NULL;

	delete[] pyramid_weights;
	pyramid_weights=NULL;
}

bool CSimpleLocalityImprovedCharKernel::load_init(FILE* src)
{
	return false;
}

bool CSimpleLocalityImprovedCharKernel::save_init(FILE* dest)
{
	return false;
}
  
static void ASSERT2 (const INT ok, const CHAR* const msg)
{
   if (! ok) {
      CIO::message(M_ERROR, msg );
   }
}

static DREAL dot_pyr (const CHAR* const x1,
		     const CHAR* const x2,
		     const INT NOF_NTS,
		     const INT NTWIDTH,
		     const INT DEGREE1,
		     const INT DEGREE2, 
		     CHAR *stage1,
		     DREAL *pyra)
{
  const INT PYRAL = 2 * NTWIDTH - 1; // total window length
  INT pyra_len, pyra_len2 ;
  DREAL pot, PYRAL_pot;
  DREAL sum;
  INT DEGREE1_1=(DEGREE1 & 0x1)==0 ;
  INT DEGREE1_1n=(DEGREE1 & ~0x1)!=0 ;
  INT DEGREE1_2=(DEGREE1 & 0x2)!=0 ;
  INT DEGREE1_3=(DEGREE1 & ~0x3)!=0 ;
  INT DEGREE1_4=(DEGREE1 & 0x4)!=0 ;
  {
    DREAL PYRAL_ = PYRAL ;
    PYRAL_pot = DEGREE1_1 ? 1.0 : PYRAL_;
    if (DEGREE1_1n) {
      PYRAL_ *= PYRAL_;
      if (DEGREE1_2) PYRAL_pot *= PYRAL_;
      if (DEGREE1_3) {
	PYRAL_ *= PYRAL_ ;
	if (DEGREE1_4) PYRAL_pot *= PYRAL_;
      }
    }
  } 
  
  ASSERT2 ((DEGREE1 & ~0x7) == 0, "DEGREE1");
  ASSERT2 ((DEGREE2 & ~0x7) == 0, "DEGREE2");
  
  pyra_len  = NOF_NTS - PYRAL + 1 ;
  pyra_len2 = (int) pyra_len / 2 ;
  {
    INT j ;
    for (j = 0; j < pyra_len; j++) 
      pyra[j] = 4*((DREAL)((j < pyra_len2) ? j+1 : pyra_len-j))/((DREAL)pyra_len) ;
    for (j = 0; j < pyra_len; j++) 
      pyra[j] /= PYRAL_pot ;
  } 
  
  /* LOOP */
  register INT conv ;
  register INT i;
  register INT j;
  for (i = 0; i < NOF_NTS; i++) {
    stage1[i] = (x1[i] == x2[i]) ;
  }
  
  sum = 0.0;
  conv = 0;
  for (j = 0; j < PYRAL; j++) 
    conv += stage1[j] ;
  for (i = 0; i < NOF_NTS-PYRAL+1; i++) {
    register DREAL pot2 ;
    if (i>0)
      conv += stage1[i+PYRAL-1] - stage1[i-1] ;
    { /* potencing of conv -- double is faster*/
      register DREAL conv2 = conv ;
      pot2 = (DEGREE1_1) ? 1.0 : conv2;
      if (DEGREE1_1n) {
	conv2 *= conv2;
	if (DEGREE1_2) pot2 *= conv2;
	if (DEGREE1_3 && DEGREE1_4) pot2 *= conv2*conv2 ;
      }
    } 
    sum += pot2*pyra[i] ;
  }
  
  pot = ((DEGREE2 & 0x1) == 0) ? 1.0 : sum;
  if ((DEGREE2 & ~0x1) != 0) {
    sum *= sum;
    if ((DEGREE2 & 0x2) != 0) pot *= sum;
    if ((DEGREE2 & ~0x3) != 0) {
      sum *= sum;
      if ((DEGREE2 & 0x4) != 0) pot *= sum;
    }
  }

  return pot ;
}

DREAL CSimpleLocalityImprovedCharKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  ASSERT(alen==blen);

  //  CIO::message("start: %ld %ld", avec, bvec) ;
  DREAL dpt ; 
  //fprintf(stderr, "length=%i, alen=%ld, id=%i, od=%i\n", length, alen, inner_degree, outer_degree) ;
  
  dpt = dot_pyr (avec, bvec,
		 alen,
		 length,
		 inner_degree,
		 outer_degree, 
		 match, pyramid_weights) ;
  dpt = dpt / pow((double)alen, (double)outer_degree) ;
  //CIO::message("end") ;
  //fprintf(stderr, "dpt=%f\n", dpt) ;

  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return (DREAL) dpt;
}
