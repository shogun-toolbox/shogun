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

#include "lib/config.h"

#include "lib/common.h"
#include "base/Parameter.h"
#include "machine/OnlineLinearMachine.h"

namespace shogun
{
/** @brief Class implementing a purely online version of LibLinear,
 * using the L2R_L1LOSS_SVC_DUAL solver only. */
class COnlineLibLinear : public COnlineLinearMachine
{
public:
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
		 * Train classifier
		 *
		 * @param data Training data, can be avoided if already
		 * initialized with it
		 *
		 * @return Whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/**
		 * Set constant Cn
		 *
		 * @param Cn_ new Cn value
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
};
}
#endif // _ONLINELIBLINEAR_H__
