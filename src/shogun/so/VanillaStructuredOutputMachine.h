/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _VANILLA_STRUCTUREDOUTPUT_MACHINE_H__
#define _VANILLA_STRUCTUREDOUTPUT_MACHINE_H__

#include <shogun/machine/LinearStructuredOutputMachine.h>

namespace shogun
{

/** 
 * @brief Class VanillaStructuredOutputMachine that implements the optimization
 * algorithm for structured output (SO) problems presented in [1] for SVM
 * learning. The optimization problem is solved using a cutting plane algorithm.
 *
 * [1] Tsochantaridis, I., Hofmann, T., Joachims, T., Altun, Y.
 *     Support Vector Machine Learning for Interdependent and Structured Ouput
 *     Spaces.
 *     http://www.cs.cornell.edu/People/tj/publications/tsochantaridis_etal_04a.pdf
 */
class CVanillaStructuredOutputMachine : public CLinearStructuredOutputMachine
{
	public:
		/** default constructor */
		CVanillaStructuredOutputMachine();

		/** standard constructor
		 *
		 * @param model structured model with application specific functions
		 * @param loss structured loss function
		 * @param labs structured labels
		 * @param features features
		 */
		CVanillaStructuredOutputMachine(CStructuredModel* model, CStructuredLoss* loss, CStructuredLabels* labs, CFeatures* features);

		/** destructor */
		~CVanillaStructuredOutputMachine();

	protected:
		/** train Vanilla SO-SVM
		 *
		 * @param data training data
		 * @return whether the training was successful
		 */
		bool train_machine(CFeatures* data = NULL);

	private:
		/** register class members */
		void register_parameters();

		/** computes the result of TODO equation
		 *
		 * @param result CResultSet structure with any argmax output
		 * @return result of the operation
		 */
		float64_t compute_loss_arg(CResultSet* result) const;

		/** insert element in the list of argmax results
		 *
		 * @param result_list list of CResultSet
		 * @param result element to insert in the list
		 */
		bool insert_result(CList* result_list, CResultSet* result) const;

	private:
		/** weight vector */
		SGVector< float64_t > m_w;

}; /* class CVanillaStructuredOutputMachine */

} /* namespace shogun */

#endif /* _VANILLA_STRUCTUREDOUTPUT_MACHINE_H__ */
