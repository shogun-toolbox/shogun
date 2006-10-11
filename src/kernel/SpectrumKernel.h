#ifndef _SPECTRUM_KERNEL_H__
#define _SPECTRUM_KERNEL_H__

#include "lib/suffixarray/DataType.h"
#include "lib/suffixarray/ErrorCode.h"
#include "lib/suffixarray/ESA.h"
#include "lib/suffixarray/I_SAFactory.h"
#include "lib/suffixarray/I_LCPFactory.h"
#include "lib/suffixarray/W_msufsort.h"
#include "lib/suffixarray/W_kasai_lcp.h"
#include "lib/suffixarray/I_WeightFactory.h"
#include "lib/suffixarray/ConstantWeight.h"
#include "lib/suffixarray/ExpDecayWeight.h"
#include "lib/suffixarray/BoundedRangeWeight.h"
#include "lib/suffixarray/KSpectrumWeight.h"

//class CSpectrumKernel : public CStringKernel<CHAR>
//{
//
//public:
//	/// Constructors
//	CSpectrumKernel(INT cachesize);
//
//	/// Destructor
//	virtual ~SpectrumKernel();
//
//	virtual void cleanup();
//
//protected:
//	////' Given contructed suffix array
//	//StringKernel(ESA *esa_);
//
//	////' Given text, build suffix array for it
//	//StringKernel(const UInt32 &size, SYMBOL *text, int verb=INFO);
//
//	///// Precompute the contribution of each intervals (or internal nodes)
//	//ErrorCode PrecomputeVal();
//	
//	/// Compute Kernel matrix
//	ErrorCode Compute_K(SYMBOL *xprime, const UInt32 &xprime_len, Real &value);
//
//	/// Set leaves array, lvs[]
//	ErrorCode Set_Lvs(const Real *leafWeight, const UInt32 *len, const UInt32 &m);
//
//	/// Set leaves array as lvs[i]=i for i=0 to esa->length
//	ErrorCode Set_Lvs();
//
//protected:
//	/// Variables
//	ESA				  *esa;
//	I_WeightFactory	  *weigher;
//	Real              *val;  //' val array. Storing precomputed val(t) values.
//	Real			  *lvs;  //' leaves array. Storing weights for leaves.
//
// private:
//	/// An iterative auxiliary function used in PrecomputeVal()
//	ErrorCode IterativeCompute(const UInt32 &left, const UInt32 &right);
//
//};
#endif
