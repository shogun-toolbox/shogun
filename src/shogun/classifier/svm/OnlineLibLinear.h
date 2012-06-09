/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2010 Soeren Sonnenburg
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (c) 2007-2009 The LIBLINEAR Project.
 * Copyright (C) 2007-2010 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _ONLINELIBLINEAR_H__
#define _ONLINELIBLINEAR_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/base/Parameter.h>
#include <shogun/machine/OnlineLinearMachine.h>

namespace shogun
{
/** @brief Class implementing a purely online version of LibLinear,
 * using the L2R_L1LOSS_SVC_DUAL solver only. */
class COnlineLibLinear : public COnlineLinearMachine
{
public:
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** Default constructor */
		COnlineLibLinear();

		/**
		 * Constructor
		 *
		 * @param C Cost constant C
		 */
		COnlineLibLinear(float64_t C);

		/**
		 * Constructor
		 *
		 * @param C Cost constant C
		 * @param traindat Training examples
		 */
		COnlineLibLinear(float64_t C, CStreamingDotFeatures* traindat);

		/** Destructor */
		virtual ~COnlineLibLinear();

		/**
		 * Set C1 and C2 constants
		 *
		 * @param c_neg C1 value
		 * @param c_pos C2 value
		 */
		virtual void set_C(float64_t c_neg, float64_t c_pos) { C1=c_neg; C2=c_pos; }

		/**
		 * Get constant C1
		 *
		 * @return C1
		 */
		virtual float64_t get_C1() { return C1; }

		/**
		 * Get constant C2
		 *
		 * @return C2
		 */
		float64_t get_C2() { return C2; }

		/**
		 * Set whether to use bias or not
		 *
		 * @param enable_bias true if bias should be used
		 */
		virtual void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }

		/**
		 * Check if bias is enabled
		 *
		 * @return If bias is enabled
		 */
		virtual bool get_bias_enabled() { return use_bias; }

		/** @return Object name */
		inline virtual const char* get_name() const { return "OnlineLibLinear"; }

		/** start training */
		virtual void start_train();

		/** stop training */
		virtual void stop_train();

		/** train on one example
		 * @param feature the feature object containing the current example. Note that get_next_example
		 *        is already called so relevalent methods like dot() and dense_dot() can be directly 
		 *        called. WARN: this function should only process ONE example, and get_next_example() 
		 *        should NEVER be called here. Use the label passed in the 2nd parameter, instead of 
		 *		  get_label() from feature, because sometimes the features might not have associated
		 *		  labels or the caller might want to provide some other labels.
		 * @param label label of this example
		 */
		virtual void train_example(CStreamingDotFeatures *feature, float64_t label);
protected:

		/**
		 * Train classifier
		 *
		 * @param data Training data, can be avoided if already
		 * initialized with it
		 *
		 * @return Whether training was successful
		 */
		//virtual bool train_machine(CFeatures* data=NULL);

private:
		/** Set up parameters */
		void init();

private:
		/// use bias or not
		bool use_bias;
		/// C1 value
		float64_t C1;
		/// C2 value
		float64_t C2;

protected:
		//========================================
		// "local" variables used during training

		float64_t C, d, G;
		float64_t QD;

		// y and alpha for example being processed
		int32_t y_current;
		float64_t alpha_current;

		// Cost constants
		float64_t Cp;
		float64_t Cn;

		// PG: projected gradient, for shrinking and stopping
		float64_t PG;
		float64_t PGmax_old;
		float64_t PGmin_old;
		float64_t PGmax_new;
		float64_t PGmin_new;

		// Diag is probably unnecessary
		float64_t diag[3];
		float64_t upper_bound[3];

		// Objective value = v/2
		float64_t v;
		// Number of support vectors
		int32_t nSV;
};
}
#endif // _ONLINELIBLINEAR_H__
