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

#ifndef __GUIHMM__H
#define __GUIHMM__H

#include <lib/config.h>
#include <base/SGObject.h>
#include <distributions/HMM.h>
#include <labels/Labels.h>
#include <labels/RegressionLabels.h>

namespace shogun
{
class CSGInterface;

/** @brief UI HMM (Hidden Markov Model) */
class CGUIHMM : public CSGObject
{
	public:
		/** constructor */
		CGUIHMM() { };
		/** constructor
		 * @param interface
		 */
		CGUIHMM(CSGInterface* interface);
		/** destructor */
		~CGUIHMM();

		/** create new HMM */
		bool new_hmm(int32_t n, int32_t m);
		/** load HMM from file */
		bool load(char* filename);
		/** save HMM to file */
		bool save(char* filename, bool is_binary=false);

		/** set num hmm tables
		 * @param param
		 */
		bool set_num_hmm_tables(char* param) ;
		/** train Baum-Welch */
		bool baum_welch_train();
		/** train Baum-Welch trans */
		bool baum_welch_trans_train();
		/** train Baum-Welch defined */
		bool baum_welch_train_defined();
		/** train Viterbi defined */
		bool viterbi_train_defined();
		/** train Viterbi */
		bool viterbi_train();
		/** linear train
		 * @param align
		 */
		bool linear_train(char align='l');
		/** linear train from file
		 * @param param
		 */
		bool linear_train_from_file(char* param);
		/** append HMM/model for CmdlineInterface */
		bool append_model(char* filename, int32_t base1=-1, int32_t base2=-1);
		/** add states to HMM */
		bool add_states(int32_t num_states=1, float64_t value=0);
		/** set HMM as POS/NEG/TEST */
		bool set_hmm_as(char* target);
		/** set HMM pseudo */
		bool set_pseudo(float64_t pseudo);
		/** set convergence criteria */
		bool convergence_criteria(
			int32_t num_iterations=100, float64_t epsilon=0.001);
		/** output HMM */
		bool output_hmm();
		/** output HMM defined */
		bool output_hmm_defined();
		/** print best path */
		bool best_path(int32_t from=0, int32_t to=100);
		/** normalize
		 * @param keep_dead_states
		 */
		bool normalize(bool keep_dead_states=false);
		/** save path
		 * @param filename
		 * @param is_binary
		 */
		bool save_path(char* filename, bool is_binary=false);
		/** save HMM likelihood  to file */
		bool save_likelihood(char* filename, bool is_binary=false);
		/** load definitions
		 * @param filename
		 * @param do_init
		 */
		bool load_definitions(char* filename, bool do_init=false);
		/** set max dim
		 * @param param
		 */
		bool set_max_dim(char* param);
		/** HMM likelihood */
		bool likelihood();
		/** chop HMM */
		bool chop(float64_t value);
		/** relative entropy
		 * @param values
		 * @param len
		 */
		bool relative_entropy(float64_t*& values, int32_t& len);
		/** entropy
		 * @param values
		 * @param len
		 */
		bool entropy(float64_t*& values, int32_t& len);
		/** define permutation entropy */
		bool permutation_entropy(int32_t width=0, int32_t seq_num=-1);
		/** get pos */
		inline CHMM* get_pos() { return pos; }
		/** get neg */
		inline CHMM* get_neg() { return neg; }
		/** get test */
		inline CHMM* get_test() { return test; }
		/** set current
		 * @param h
		 */
		inline void set_current(CHMM* h) { working=h; }
		/** get current */
		inline CHMM* get_current() { return working; }
		/** get pseudo */
		inline float64_t get_pseudo() { return PSEUDO; }

		/** classify
		 * @param output
		 */
		CRegressionLabels* classify(CRegressionLabels* output=NULL);
		/** classify example
		 * @param idx
		 */
		float64_t classify_example(int32_t idx);
		/** one class classify
		 * @param output
		 */
		CRegressionLabels* one_class_classify(CRegressionLabels* output=NULL);
		/** linear one class classify
		 * @param output
		 */
		CRegressionLabels* linear_one_class_classify(CRegressionLabels* output=NULL);
		/** one class classfiy example
		 * @param idx
		 */
		float64_t one_class_classify_example(int32_t idx);

		/** @return object name */
		virtual const char* get_name() const { return "GUIHMM"; }

	protected:
		/** converge
		 * @param x
		 * @param y
		 */
		bool converge(float64_t x, float64_t y);
		/** switch model
		 * @param m1
		 * @param m2
		 */
		void switch_model(CHMM** m1, CHMM** m2);

		/** working */
		CHMM* working;

		/** pos */
		CHMM* pos;
		/** neg */
		CHMM* neg;
		/** test */
		CHMM* test;

		/** pseudo */
		float64_t PSEUDO;
		/** M */
		int32_t M;

	protected:
		/** ui */
		CSGInterface* ui;
};
}
#endif
