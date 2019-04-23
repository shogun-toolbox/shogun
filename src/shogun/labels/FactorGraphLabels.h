/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shell Hu, Fernando Iglesias, Yuyu Zhang, Bjoern Esser
 */

#ifndef __FACTORGRAPH_LABELS_H__
#define __FACTORGRAPH_LABELS_H__

#include <shogun/lib/config.h>

#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/StructuredData.h>
#include <shogun/lib/StructuredDataTypes.h>

namespace shogun
{

class FactorGraphLabels;

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
	static std::shared_ptr<FactorGraphObservation> obtain_from_generic(std::shared_ptr<StructuredData> base_data)
	{
		if ( base_data->get_structured_data_type() == SDT_FACTOR_GRAPH )
			return std::static_pointer_cast<FactorGraphObservation>(base_data);
		else
			error("base_data must be of dynamic type FactorGraphObservation");

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
 * a DynamicObjectArray. */
class FactorGraphLabels : public StructuredLabels
{
	public:
		/** default constructor */
		FactorGraphLabels();

		/** standard constructor
		 *
		 * @param num_labels number of labels
		 */
		FactorGraphLabels(int32_t num_labels);

		/** destructor */
		virtual ~FactorGraphLabels();

		/** @return object name */
		virtual const char* get_name() const { return "FactorGraphLabels"; }

	private:
		/** internal initialization */
		void init();

}; /* FactorGraphLabels */

} /* namespace shogun */

#endif /* __FACTORGRAPH_LABELS_H__ */
