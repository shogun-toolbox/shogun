/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Roman Votyakov, Soeren Sonnenburg, Heiko Strathmann,
 *          Yuyu Zhang, Ariane Paola Gomes
 */

#ifndef CGRADIENTRESULT_H_
#define CGRADIENTRESULT_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/EvaluationResult.h>
#include <shogun/lib/Map.h>
#include <shogun/base/Parameter.h>

namespace shogun
{

/** @brief Container class that returns results from GradientEvaluation. It
 * contains the function value as well as its gradient.
 */
class GradientResult : public EvaluationResult
{
public:
	/** default constructor */
	GradientResult() : EvaluationResult()
	{
		m_total_variables=0;
		m_gradient=NULL;
		m_parameter_dictionary=NULL;
	}

	virtual ~GradientResult()
	{


	}

	/** returns the name of the evaluation result
	 *
	 *  @return name GradientResult
	 */
	virtual const char* get_name() const { return "GradientResult"; }

	/** prints the function value and gradient contained in the object */
	virtual void print_result()
	{
		REQUIRE(m_gradient, "Gradient map should not be NULL\n")
		REQUIRE(m_parameter_dictionary, "Parameter dictionary should not be "
				"NULL\n")

		// print value of the function
		SG_SPRINT("Value: [")

		for (index_t i=0; i<m_value.vlen-1; i++)
			SG_SPRINT("%f, ", m_value[i])

		if (m_value.vlen>0)
			SG_SPRINT("%f", m_value[m_value.vlen-1])

		SG_SPRINT("] ")

		// print gradient wrt parameters
		SG_SPRINT("Gradient: [")

		for (index_t i=0; i<m_gradient->get_num_elements(); i++)
		{
			auto param_node=
				m_gradient->get_node_ptr(i);

			// get parameter name
			const char* param_name=param_node->key->m_name;

			// get object name
			const char* object_name=
				m_parameter_dictionary->get_element(param_node->key)->get_name();

			// get gradient wrt parameter
			SGVector<float64_t> param_gradient=param_node->data;

			SG_PRINT("%s.%s: ", object_name, param_name)

			for (index_t j=0; j<param_gradient.vlen-1; j++)
				SG_SPRINT("%f, ", param_gradient[j])

			if (i==m_gradient->get_num_elements()-1)
			{
				if (param_gradient.vlen>0)
					SG_PRINT("%f", param_gradient[param_gradient.vlen-1])
			}
			else
			{
				if (param_gradient.vlen>0)
					SG_PRINT("%f; ", param_gradient[param_gradient.vlen-1])
			}
		}

		SG_SPRINT("] Total Variables: %u\n", m_total_variables)
	}

	/** return number of total variables in gradient map
	 *
	 * @return number of total variables
	 */
	virtual uint32_t get_total_variables()
	{
		return m_total_variables;
	}

	/** sets value of the function
	 *
	 * @param value value of the function
	 */
	virtual void set_value(SGVector<float64_t> value)
	{
		m_value=SGVector<float64_t>(value);
	}

	/** returns value of the function
	 *
	 * @return value of the function
	 */
	virtual SGVector<float64_t> get_value()
	{
		return SGVector<float64_t>(m_value);
	}

	/** sets gradient map
	 *
	 * @param gradient gradient map to set
	 */
	virtual void set_gradient(std::shared_ptr<CMap<TParameter*, SGVector<float64_t> >> gradient)
	{
		REQUIRE(gradient, "Gradient map should not be NULL\n")



		m_gradient=gradient;

		m_total_variables=0;

		for (index_t i=0; i<gradient->get_num_elements(); i++)
		{
			CMapNode<TParameter*, SGVector<float64_t> >* node=
				m_gradient->get_node_ptr(i);
			m_total_variables+=node->data.vlen;
		}
	}

	/** returns gradient map
	 *
	 * @return gradient map
	 */
	virtual std::shared_ptr<CMap<TParameter*, SGVector<float64_t> >> get_gradient()
	{

		return m_gradient;
	}

	/** sets parameter dictionary
	 *
	 * @param parameter_dictionary parameter dictionary
	 */
	virtual void set_paramter_dictionary(
			std::shared_ptr<CMap<TParameter*, SGObject*>> parameter_dictionary)
	{


		m_parameter_dictionary=parameter_dictionary;
	}

	/** returns parameter dictionary
	 *
	 * @return parameter dictionary
	 */
	virtual std::shared_ptr<CMap<TParameter*, SGObject*>> get_paramter_dictionary()
	{

		return m_parameter_dictionary;
	}

private:
	/** function value */
	SGVector<float64_t> m_value;

	/** function gradient */
	std::shared_ptr<CMap<TParameter*, SGVector<float64_t> >> m_gradient;

	/** which objects do the gradient parameters belong to? */
	std::shared_ptr<CMap<TParameter*, SGObject*>>  m_parameter_dictionary;

	/** total number of variables represented by the gradient */
	uint32_t m_total_variables;
};
}
#endif /* CGRADIENTRESULT_H_ */
