/*
 * This is a very preliminary head file for GMM EM training.
 * Since my proposal topic is GMM training with carefully implemented EM algorithm,
 * I define the basic data structure and basic training function in this headfile
 */
 
#ifndef __GMM_H__
#define __GMM_H__

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
	// Constructor. Initializes all variables and structures.
	GMMModel();
	/// Destructor - cleans up
	virtual ~GMMModel();

	// GMM K-means function. K-means is used for mixture initilization
	GMMKMeans();
	
	// GMM EM training algorithm
	GMMEM();
	
	// remove the singular mixtures
	GMMRemoveSingular();
	
	// split mixtures with large weights in order to obtain desired mixture number
	GMMSplitMix();
	
	// Singular Value Decomposition of Full Covariance Matrix
	GMMSVDCov();
	
	// calculate likelihood given a feature points
	float64 GMMprob( float64_t* );
	
	// calculate posterior probability of a featur point corresponding to each mixture
	float64* GMMposteriorprob( float64_t* );
	
	// write down GMM model
	GMMWriteOut();
	
	// readin existing GMM
	GMMReadIn();
	
	protected:
		
	struct GMM_basic_config // basic configuration
	{
		int32_t	nMixNum;			// mixture number
		int32_t	nVecSize;			// feature dimension
		int32_t niter;				// EM iteration number
		bool bDiagCov;				// if bDiagCov = 1, use Diagonal Covariances.
													//if bDiagCov = 0, use full covariances, and then SVD is used as proposed in the proposal.
	}
	
	struct GMM_Model // basic configuration
	{
		float64_t* pWeights;			// buffer for GMM weights
		float64_t* pMeans;			  // buffer for GMM means
		// three different ways for storing GMM covariances
		float64_t* pDiagCov;		// buffer for GMM diagonal Covariances (if bDiagCov = 1)
		float64_t* pFullCov;		// buffer for GMM full Covariances  (if bDiagCov = 0)
		float64_t* pSVDLeft;		// buffer for SVD Left matrix, which is used for full Covariances.
		float64_t* pSVDRight;		// buffer for SVD Right matrix, which is used for full Covariances.
		float64_t* pSVDDiag;		// buffer for SVD singular values, which is used for full Covariances.		
	}
	
	int32_t nFeatures;    // feature points number
	float64_t* pFeatureProbs; // buffer for storing likelihoods of each feature point corresponding to each mixture.
														// size of pFeatureProbs should be nFeatures*nMixNum
	float64_t *pMeanprob;			// buffer for storing first order GMM statistics
	float64_t *pCovprob;			// buffer for storing second order GMM statistics
	
	float64_t fpFloor;				// floor for posterior prob, in order to avoid log(0)
	float64_t fwFloor;				// mixture weights floor, may not be used if we decide to determine singular mixtures by their weights
	float64_t fcFloor;				// covariance floor
}
#endif