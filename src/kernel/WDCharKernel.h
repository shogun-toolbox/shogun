#ifndef _WDCHARKERNEL_H___
#define _WDCHARKERNEL_H___

#include "lib/common.h"
#include "kernel/CharKernel.h"
#include "kernel/WeightedDegreeCharKernel.h"

enum EWDKernType
{
	E_WD=0,
	E_CONST=1,
	E_LINEAR=2,
	E_SQPOLY=3,
	E_CUBICPOLY=4,
	E_EXP=5,
	E_LOG=6
};

class CWDCharKernel: public CCharKernel
{
	public:
		CWDCharKernel(LONG size, EWDKernType type, INT degree);
		~CWDCharKernel();

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
		virtual void cleanup();

		/// load and save kernel init_data
		bool load_init(FILE* src);
		bool save_init(FILE* dest);

		// return what type of kernel we are Linear,Polynomial, Gaussian,...
		virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREE; }

		// return the name of a kernel
		virtual const CHAR* get_name() { return "WD"; }
	protected:
		bool init_matching_weights();

		bool init_matching_weights_wd();
		bool init_matching_weights_const();
		bool init_matching_weights_linear();
		bool init_matching_weights_sqpoly();
		bool init_matching_weights_cubicpoly();
		bool init_matching_weights_exp();
		bool init_matching_weights_log();

		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		REAL compute(INT idx_a, INT idx_b);
		/*    compute_kernel*/

		virtual void remove_lhs();
		virtual void remove_rhs();

	protected:
		REAL* matching_weights;

		INT degree;
		INT seq_length;

		EWDKernType type;

		double* sqrtdiag_lhs;
		double* sqrtdiag_rhs;

		bool initialized;
		bool* match_vector;
};
#endif
