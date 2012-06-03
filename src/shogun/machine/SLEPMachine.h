/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  SLEPMACHINE_H_
#define  SLEPMACHINE_H_

#include <shogun/lib/config.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
/** @brief  */
class CSLEPMachine : public CLinearMachine
{
	public:

		/** default constructor */
		CSLEPMachine();

		/** constructor
		 *
		 * @param z regularization coefficient
		 * @param training_data training features
		 * @param training_labels training labels
		 */
		CSLEPMachine(
		     float64_t z, CDotFeatures* training_data, 
		     CLabels* training_labels);

		/** destructor */
		virtual ~CSLEPMachine();

		/** get name */
		virtual const char* get_name() const 
		{
			return "SLEPMachine";
		}

		int32_t get_max_iter() const;
		int32_t get_regularization() const;
		int32_t get_termination() const;
		float64_t get_tolerance() const;
		float64_t get_z() const;

		void set_max_iter(int32_t max_iter);
		void set_regularization(int32_t regularization);
		void set_termination(int32_t termination);
		void set_tolerance(float64_t tolerance);
		void set_z(float64_t z);

	private:

		/** register parameters */
		void register_parameters();

	protected:

		int32_t m_regularization;

		int32_t m_termination;

		int32_t m_max_iter;

		float64_t m_tolerance;

		/** regularization coefficient */
		float64_t m_z;

};
}
#endif   /* ----- #ifndef SLEPMACHINE_H_  ----- */
