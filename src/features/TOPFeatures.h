#ifndef _CTOPFEATURES__H__
#define _CTOPFEATURES__H__

#include "features/RealFeatures.h"
#include "hmm/HMM.h"

class CTOPFeatures: public CRealFeatures
{
 public:
		CTOPFeatures(CHMM* p, CHMM* n);
		~CTOPFeatures();
	    virtual REAL* set_feature_matrix();
 protected:
		virtual REAL* compute_feature_vector(int num, int& len);
		
		/// computes the featurevector to the address addr
		void compute_feature_vector(REAL* addr, int num, int& len);

 protected:
		CHMM* pos;
		CHMM* neg;
};
#endif
