#ifndef _CTOPFEATURES__H__
#define _CTOPFEATURES__H__

#include "features/RealFeatures.h"
#include "distributions/hmm/HMM.h"

class CTOPFeatures: public CRealFeatures
{
	struct T_HMM_INDIZES
	{
		INT* idx_p;
		INT* idx_q;
		INT* idx_a_rows;
		INT* idx_a_cols;
		INT* idx_b_rows;
		INT* idx_b_cols;

		INT num_p;
		INT num_q;
		INT num_a;
		INT num_b;
	};

	public:
	CTOPFeatures(LONG size, CHMM* p, CHMM* n, bool neglin, bool poslin);
	CTOPFeatures(const CTOPFeatures &orig);

	virtual ~CTOPFeatures();

	void set_models(CHMM* p, CHMM* n);
	virtual REAL* set_feature_matrix();

	virtual CFeatures* duplicate() const;

	INT compute_num_features();

	bool compute_relevant_indizes(CHMM* hmm, T_HMM_INDIZES* hmm_idx);

	protected:
	virtual REAL* compute_feature_vector(INT num, INT& len, REAL* target=NULL);

	/// computes the featurevector to the address addr
	void compute_feature_vector(REAL* addr, INT num, INT& len);

	protected:
	CHMM* pos;
	CHMM* neg;
	bool neglinear;
	bool poslinear;

	T_HMM_INDIZES pos_relevant_indizes;
	T_HMM_INDIZES neg_relevant_indizes;
};
#endif
