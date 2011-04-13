/*
 * This is a very preliminary head file for GMM EM training.
 * Since my proposal topic is GMM training with carefully implemented EM algorithm,
 * I define the basic data structure and basic training function in this headfile
 */
 
#ifndef __GMM_H__
#define __GMM_H__

#ifndef PI
/** just in case PI is not defined */
#define PI   3.14159265358979
#endif

/** 2*PI */
#define DPI  6.28318530717959

/** log(0). Any log value < logzero is set to logzero */
#define LogZero (-1.0E10)

/** lowest exp() arg  = log(MinLogArg) */
#define MinExpArg (-700.0)

/** lowest log() arg  = exp(MinExpArg) */
#define MinLogArg 9.8597E-305

#include <stdio.h>

#include "lib/Mathematics.h"
#include "lib/common.h"
#include "lib/io.h"
#include "lib/config.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "distributions/Distribution.h"

namespace shogun
{
class GMMModel
{
	public:
	/** Constructor. Initializes all variables and structures. */
	GMMModel();
	/** Destructor - cleans up */
	virtual ~GMMModel();

	/** GMM K-means function. K-means is used for mixture initilization. */
	GMMKMeans();
	
	/** GMM EM training algorithm */
	GMMEM();
	
	/** remove the singular mixtures */
	GMMRemoveSingular();
	
	/** split mixtures with large weights in order to obtain desired mixture number. */
	GMMSplitMix();
	
	/** Singular Value Decomposition of Full Covariance Matrix. */
	GMMSVDCov();
	
	/** Calculate likelihood given a feature points. */
	float64 GMMprob( float64_t* );
	
	/** calculate posterior probability of a featur point corresponding to each mixture. */
	float64* GMMposteriorprob( float64_t* );
	
	/** write down GMM model. */
	GMMWriteOut();
	
	/** readin existing GMM. */
	GMMReadIn();
	
	/** GMM mixture constants calculation. */
	GMMMixConst();
	
	protected:
	/** basic configuration */
	struct GMM_basic_config
	{
		/** mixture number */
		int32_t	nMixNum;	
		
		/** feature dimension	*/
		int32_t	nVecSize;
		
		/** EM iteration number */
		int32_t niter;
		
		/** if bDiagCov = 1, use Diagonal Covariances.
		 *  if bDiagCov = 0, use full covariances, and then SVD is used as proposed in the proposal.
		 */
		bool bDiagCov;									
	}
	
	/** basic configuration */
	struct GMM_Model 
	{
		/** buffer for GMM mixture consts
		 *  for each mixture, const=-0.5*(nVecSize*log(2*Pi)+log(det(Cov)))
		 */
		float64_t* fMixConst;
		
		/** buffer for GMM weights */
		float64_t* pWeights;
		
		/** buffer for GMM means */
		float64_t* pMeans;
		
		/** three different ways for storing GMM covariances */
		/** buffer for GMM diagonal Covariances (if bDiagCov = 1) */
		float64_t* pDiagCov;
		
		/** buffer for GMM full Covariances  (if bDiagCov = 0) */
		float64_t* pFullCov;
		
		/** buffer for SVD Left matrix, which is used for full Covariances.	 */
		float64_t* pSVDLeft;
		
		/** buffer for SVD Right matrix, which is used for full Covariances. */
		float64_t* pSVDRight;
		
		/** buffer for SVD singular values, which is used for full Covariances. */	
		float64_t* pSVDDiag;
	}
	/** feature points number */
	int32_t nFeatures;
	
  /** buffer for storing likelihoods of each feature point corresponding to each mixture.
	 * size of pFeatureProbs should be nFeatures*nMixNum
	 */
	float64_t* pFeatureProbs;
	
	/** buffer for storing first order GMM statistics */					
	float64_t *pMeanprob;
	
  /** buffer for storing second order GMM statistics */
	float64_t *pCovprob;
	
	/** floor for posterior prob, in order to avoid log(0) */
	float64_t fpFloor;
	
	/** mixture weights floor, may not be used if we decide to determine singular mixtures by their weights */
	float64_t fwFloor;
	
	/** covariance floor */
	float64_t fcFloor;
}
#endif