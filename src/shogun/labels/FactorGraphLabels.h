/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#ifndef __FACTORGRAPH_LABELS_H__
#define __FACTORGRAPH_LABELS_H__

#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/StructuredData.h>
#include <shogun/lib/StructuredDataTypes.h>

namespace shogun
{

class CFactorGraphLabels;

/** @brief Class FactorGraphObservation is used as
 * the structured output */
class FactorGraphObservation : public StructuredData
{
public:
	/** data type */
	STRUCTURED_DATA_TYPE(SDT_FACTOR_GRAPH);

	/** default constructor */
	FactorGraphObservation() : StructuredData() { }

	/** constructor discrete labeling observation
	 *
	 * @param observed_state Discrete labeling of a set of variables.
	 * @param loss_weights weighted loss for each variable
	 */
	FactorGraphObservation(SGVector<int32_t> observed_state,
		SGVector<float64_t> loss_weights);

	~FactorGraphObservation() { }

	/** helper method used to specialize a base class instance
	 *
	 * @param base_data its dynamic type must be FactorGraphObservation
	 */
	static FactorGraphObservation* obtain_from_generic(StructuredData* base_data)
	{
		if ( base_data->get_structured_data_type() == SDT_FACTOR_GRAPH )
			return (FactorGraphObservation*) base_data;
		else
			SG_SERROR("base_data must be of dynamic type FactorGraphObservation\n")

		return NULL;
	}

	/** @return name of SGSerializable */
	virtual const char* get_name() const { return "FactorGraphObservation"; }

	/** @return data */
	SGVector<int32_t> get_data() const;

	/** @return loss weights */
	SGVector<float64_t> get_loss_weights() const;

	/** set loss weights
	 *
	 * @param loss_weights weights for weighted hamming loss
	 */
	void set_loss_weights(SGVector<float64_t> loss_weights);

protected:
	/** loss weights, usually for weighted hamming loss */
	SGVector<float64_t> m_loss_weights;

	/** ground truth states */
	SGVector<int32_t> m_observed_state;
};


/** @brief Class FactorGraphLabels used e.g. in the application of Structured Output
 * (SO) learning with the FactorGraphModel. Each of the labels is represented by a
 * graph. Each label is of type FactorGraphObservation and all of them are stored in
 * a CDynamicObjectArray. */
class CFactorGraphLabels : public CStructuredLabels
{
	public:
		/** default constructor */
		CFactorGraphLabels();

		/** standard constructor
		 *
		 * @param num_labels number of labels
		 */
		CFactorGraphLabels(int32_t num_labels);

		/** destructor */
		virtual ~CFactorGraphLabels();

		/** @return object name */
		virtual const char* get_name() const { return "FactorGraphLabels"; }

	private:
		/** internal initialization */
		void init();

}; /* CFactorGraphLabels */

} /* namespace shogun */

#endif /* __FACTORGRAPH_LABELS_H__ */
