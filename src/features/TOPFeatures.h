/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CTOPFEATURES__H__
#define _CTOPFEATURES__H__

#include "features/RealFeatures.h"
#include "distributions/hmm/HMM.h"

/** HMM indices */
struct T_HMM_INDIZES
{
	/** index p */
	INT* idx_p;
	/** index q */
	INT* idx_q;
	/** index a rows */
	INT* idx_a_rows;
	/** index a cols */
	INT* idx_a_cols;
	/** index b rows */
	INT* idx_b_rows;
	/** index b cols */
	INT* idx_b_cols;

	/** number p */
	INT num_p;
	/** number q */
	INT num_q;
	/** number a */
	INT num_a;
	/** number b */
	INT num_b;
};

/** class TOPFeatures */
class CTOPFeatures: public CRealFeatures
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param p positive HMM
		 * @param n negative HMM
		 * @param neglin if negative HMM is of linear shape
		 * @param poslin if positive HMM is of linear shape
		 */
		CTOPFeatures(INT size, CHMM* p, CHMM* n, bool neglin, bool poslin);

		/** copy constructor */
		CTOPFeatures(const CTOPFeatures &orig);

		virtual ~CTOPFeatures();

		/** set HMMs
		 *
		 * @param p positive HMM
		 * @param n negative HMM
		 */
		void set_models(CHMM* p, CHMM* n);

		/** set feature matrix
		 *
		 * @return something floaty
		 */
		virtual DREAL* set_feature_matrix();

		/** compute number of features
		 *
		 * @return number of features
		 */
		INT compute_num_features();

		/** compute relevant indices
		 *
		 * @param hmm HMM to compute for
		 * @param hmm_idx HMM index
		 * @return if computing was successful
		 */
		bool compute_relevant_indizes(CHMM* hmm, T_HMM_INDIZES* hmm_idx);

	protected:
		/** compute feature vector
		 *
		 * @param num num
		 * @param len len
		 * @param target
		 * @return something floaty
		 */
		virtual DREAL* compute_feature_vector(INT num, INT& len, DREAL* target=NULL);

		/** computes the feature vector to the address addr
		 *
		 * @param addr address
		 * @param num num
		 * @param len len
		 */
		void compute_feature_vector(DREAL* addr, INT num, INT& len);

	protected:
		/** positive HMM */
		CHMM* pos;
		/** negative HMM */
		CHMM* neg;
		/** if negative HMM is a LinearHMM */
		bool neglinear;
		/** if positive HMM is a LinearHMM */
		bool poslinear;

		/** positive relevant indices */
		T_HMM_INDIZES pos_relevant_indizes;
		/** negative relevant indices */
		T_HMM_INDIZES neg_relevant_indizes;
};
#endif
