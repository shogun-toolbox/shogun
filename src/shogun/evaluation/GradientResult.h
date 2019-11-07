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
		require(!m_gradient.empty(), "Gradient map should not be NULL");
		require(!m_parameter_dictionary.empty(), "Parameter dictionary should not be "
				"NULL");

		// print value of the function
		io::print("Value: [");

		for (index_t i=0; i<m_value.vlen-1; i++)
			io::print("{}, ", m_value[i]);

		if (m_value.vlen>0)
			io::print("{}", m_value[m_value.vlen-1]);

		io::print("] ");

		// print gradient wrt parameters
		io::print("Gradient: [");
		size_t i = 0;
		for (const auto& gradient_param: m_gradient)
		{
			// get parameter name
			std::string param_name=gradient_param.first;

			// get object name
			const char* object_name=m_parameter_dictionary[gradient_param.first]->get_name();

			// get gradient wrt parameter
			SGVector<float64_t> param_gradient=gradient_param.second; 

			io::print("{}.{}: ", object_name, param_name);

			for (index_t j=0; j<param_gradient.vlen-1; j++)
				io::print("{}, ", param_gradient[j]);

			if (i==m_gradient.size()-1)
			{
				if (param_gradient.vlen>0)
					io::print("{}", param_gradient[param_gradient.vlen-1]);
			}
			else
			{
				if (param_gradient.vlen>0)
					io::print("{}; ", param_gradient[param_gradient.vlen-1]);
			}
			++i;
		}

		io::print("] Total Variables: {}\n", m_total_variables);
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
	virtual void set_gradient(const std::map<std::string, SGVector<float64_t>>& gradient)
	{
		m_gradient=gradient;

		m_total_variables=0;
		for (const auto& grad_i: gradient)
			m_total_variables += grad_i.second.size();
	}

	/** returns gradient map
	 *
	 * @return gradient map
	 */
	virtual std::map<std::string, SGVector<float64_t>> get_gradient()
	{
		return m_gradient;
	}

	/** sets parameter dictionary
	 *
	 * @param parameter_dictionary parameter dictionary
	 */
	virtual void set_parameter_dictionary(
			std::map<std::string, std::shared_ptr<SGObject>> parameter_dictionary)
	{
		m_parameter_dictionary=parameter_dictionary;
	}

	/** returns parameter dictionary
	 *
	 * @return parameter dictionary
	 */
	virtual std::map<std::string, std::shared_ptr<SGObject>> get_parameter_dictionary() 
	{
		return m_parameter_dictionary;
	}

private:
	/** function value */
	SGVector<float64_t> m_value;

	/** function gradient */
	std::map<std::string, SGVector<float64_t>> m_gradient;

	/** which objects do the gradient parameters belong to? */
	std::map<std::string, std::shared_ptr<SGObject>>  m_parameter_dictionary;

	/** total number of variables represented by the gradient */
	uint32_t m_total_variables;
};
}
#endif /* CGRADIENTRESULT_H_ */
