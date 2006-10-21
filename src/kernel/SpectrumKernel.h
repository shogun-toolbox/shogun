#ifndef _SPECTRUM_KERNEL_H__
#define _SPECTRUM_KERNEL_H__

#include "kernel/StringKernel.h"
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

class CSpectrumKernel : public CStringKernel<CHAR>
{

public:

	/// Constructors
	CSpectrumKernel(INT cachesize);
	CSpectrumKernel(CStringFeatures<CHAR>* l, CStringFeatures<CHAR>* r, INT cachesize);

	/// Destructor
	virtual ~CSpectrumKernel();

	virtual void cleanup();

protected:
	///// Precompute the contribution of each intervals (or internal nodes)
	ErrorCode PrecomputeVal();
	
	/// Compute Kernel matrix
	ErrorCode Compute_K(SYMBOL *xprime, const UInt32 &xprime_len, Real &value);

	/// Set leaves array, lvs[]
	ErrorCode Set_Lvs(const Real *leafWeight, const UInt32 *len, const UInt32 &m);

	/// Set leaves array as lvs[i]=i for i=0 to esa->length
	ErrorCode Set_Lvs();

	/// compute kernel function for strings a and b
	DREAL compute(INT idx_a, INT idx_b);

	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type() { return K_SPECTRUMKERNEL; }

	// return the name of a kernel
	virtual const CHAR* get_name() { return "SpectrumKernel" ; } ;

	virtual bool init(CFeatures* lhs, CFeatures* rhs, bool force);

	bool load_init(FILE* src);
	bool save_init(FILE* dest);

protected:
	/// Variables
	ESA				  *esa;
	I_WeightFactory	  *weigher;
	Real              *val;  //' val array. Storing precomputed val(t) values.
	Real			  *lvs;  //' leaves array. Storing weights for leaves.

 private:
	/// An iterative auxiliary function used in PrecomputeVal()
	ErrorCode IterativeCompute(const UInt32 &left, const UInt32 &right);
};
#endif
