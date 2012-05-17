/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _STRUCTUREDOUTPUTMACHINE_H__
#define _STRUCTUREDOUTPUTMACHINE_H__

#include <shogun/features/StructuredLabels.h>
#include <shogun/lib/StructuredData.h>
#include <shogun/machine/Machine.h>
#include <shogun/so/StructuredLoss.h>
#include <shogun/so/StructuredModel.h>

namespace shogun
{

/** TODO doc */
class CStructuredOutputMachine : public CMachine
{
	public:
		/** deafult constructor */
		CStructuredOutputMachine();

		/** standard constructor
		 *
		 * @param model structured model with application specific functions
		 * @param loss structured loss function
		 * @param labs structured labels
		 */
		CStructuredOutputMachine(CStructuredModel* model, CStructuredLoss* loss, CStructuredLabels* labs);

		/** destructor */
		virtual ~CStructuredOutputMachine();

		/** set labels
		 *
		 * @param labs labels
		 */
		virtual void set_labels(CStructuredLabels* labs);

		/** @return object name */
		inline virtual const char* get_name() const 
		{ 
			return "StructuredOutputMachine"; 
		}

		/** apply machine to the currently set features
		 *
		 * @return output 'labels'
		 */
		/* TODO change this to StructuredLabels when hierarchy fixed */
		virtual CLabels* apply();

		/** apply machine to data
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		/* TODO change this to StructuredLabels when hierarchy fixed */
		virtual CLabels* apply(CFeatures* data);

	private:
		/** register class members */
		void register_parameters();

	protected:
		/** the model that contains the application dependent modules */
		CStructuredModel* m_model;

		/** the general loss function */
		CStructuredLoss* m_loss;


}; /* class CStructuredOutputMachine */

} /* namespace shogun */

#endif /* _STRUCTUREDOUTPUTMACHINE_H__ */
