/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Written (W) 1999-2007 Fabio De Bona
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

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
	CTOPFeatures(INT size, CHMM* p, CHMM* n, bool neglin, bool poslin);
	CTOPFeatures(const CTOPFeatures &orig);

	virtual ~CTOPFeatures();

	void set_models(CHMM* p, CHMM* n);
	virtual DREAL* set_feature_matrix();

	INT compute_num_features();

	bool compute_relevant_indizes(CHMM* hmm, T_HMM_INDIZES* hmm_idx);

	protected:
	virtual DREAL* compute_feature_vector(INT num, INT& len, DREAL* target=NULL);

	/// computes the featurevector to the address addr
	void compute_feature_vector(DREAL* addr, INT num, INT& len);

	protected:
	CHMM* pos;
	CHMM* neg;
	bool neglinear;
	bool poslinear;

	T_HMM_INDIZES pos_relevant_indizes;
	T_HMM_INDIZES neg_relevant_indizes;
};
#endif
