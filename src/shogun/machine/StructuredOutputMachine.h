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
#include <shogun/structure/StructuredModel.h>

namespace shogun
{
class CStructuredModel;

/** TODO doc */
class CStructuredOutputMachine : public CMachine
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_STRUCTURED);

		/** deafult constructor */
		CStructuredOutputMachine();

		/** standard constructor
		 *
		 * @param model structured model with application specific functions
		 * @param labs structured labels
		 */
		CStructuredOutputMachine(CStructuredModel* model, CStructuredLabels* labs);

		/** destructor */
		virtual ~CStructuredOutputMachine();

		/** set structured model
		 *
		 * @param model structured model to set
		 */
		void set_model(CStructuredModel* model);

		/** get structured model
		 *
		 * @return structured model
		 */
		CStructuredModel* get_model() const;

		/** @return object name */
		virtual const char* get_name() const
		{ 
			return "StructuredOutputMachine";
		}

		/** set labels
		 *
		 * @param lab labels
		 */
		virtual void set_labels(CLabels* lab);

	private:
		/** register class members */
		void register_parameters();

	protected:
		/** the model that contains the application dependent modules */
		CStructuredModel* m_model;


}; /* class CStructuredOutputMachine */

} /* namespace shogun */

#endif /* _STRUCTURED_OUTPUT_MACHINE__H__ */
