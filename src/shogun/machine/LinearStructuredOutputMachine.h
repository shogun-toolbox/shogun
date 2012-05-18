/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _LINEARSTRUCTUREDOUTPUTMACHINE_H__
#define _LINEARSTRUCTUREDOUTPUTMACHINE_H__

#include <shogun/features/Features.h>
#include <shogun/machine/StructuredOutputMachine.h>

namespace shogun
{

/** TODO doc */
class CLinearStructuredOutputMachine : public CStructuredOutputMachine
{
	public:
		/** default constructor  */
		CLinearStructuredOutputMachine();

		/** standard constructor
		 *
		 * @param model structured model with application specific functions
		 * @param loss loss function
		 * @param labs structured labels
		 * @param features features
		 */
		CLinearStructuredOutputMachine(CStructuredModel* model, CLossFunction* loss, CStructuredLabels* labs, CFeatures* features);

		/** destructor */
		virtual ~CLinearStructuredOutputMachine();

		/** set features
		 *
		 * @param f features
		 */
		void set_features(CFeatures* f);

		/** get features
		 *
		 * @return features
		 */
		CFeatures* get_features() const;

		/** @return object name */
		inline virtual const char* get_name() const 
		{ 
			return "LinearStructuredOutputMachine"; 
		}

	private:
		/** register class members */
		void register_parameters();

	protected:
		/** feature vectors */
		CFeatures* m_features;

}; /* class CLinearStructuredOutputMachine */

} /* namespace shogun */

#endif /* _LINEARSTRUCTUREDOUTPUTMACHINE_H__ */
