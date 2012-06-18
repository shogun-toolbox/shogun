/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _STRUCTURED_OUTPUT_MACHINE__H__
#define _STRUCTURED_OUTPUT_MACHINE__H__

#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/StructuredData.h>
#include <shogun/machine/Machine.h>
#include <shogun/loss/LossFunction.h>
#include <shogun/structure/StructuredModel.h>

namespace shogun
{
class CStructuredModel;
class CLossFunction;

/** TODO doc */
class CStructuredOutputMachine : public CMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_STRUCTURED);

		/** deafult constructor */
		CStructuredOutputMachine();

		/** standard constructor
		 *
		 * @param model structured model with application specific functions
		 * @param loss loss function
		 * @param labs structured labels
		 */
		CStructuredOutputMachine(CStructuredModel* model, CLossFunction* loss, CStructuredLabels* labs);

		/** destructor */
		virtual ~CStructuredOutputMachine();

		/** set structured model
		 *
		 * @param model structured model to set
		 */
		void set_model(CStructuredModel* model);

		/** set loss function
		 *
		 * @param loss loss function to set
		 */
		void set_loss(CLossFunction* loss);

		/** @return object name */
		virtual const char* get_name() const 
		{ 
			return "StructuredOutputMachine"; 
		}

	private:
		/** register class members */
		void register_parameters();

	protected:
		/** the model that contains the application dependent modules */
		CStructuredModel* m_model;

		/** the general loss function */
		CLossFunction* m_loss;


}; /* class CStructuredOutputMachine */

} /* namespace shogun */

#endif /* _STRUCTURED_OUTPUT_MACHINE__H__ */
