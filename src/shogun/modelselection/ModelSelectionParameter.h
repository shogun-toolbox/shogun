/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Eleftherios Avramidis
 */

#ifndef __MODELSELECTIONPARAMETER_H_
#define __MODELSELECTIONPARAMETER_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/base/AnyParameter.h>

namespace shogun
{

class CModelSelectionParameter: public CSGObject
{
public:
	/** default constructor */
	CModelSelectionParameter();

	/** destructor */
	virtual ~CModelSelectionParameter();

	/** @return name of the SGSerializable */
	virtual const char* get_name() const { return "ModelSelectionParameter"; }

	template<typename T>
	void add_param(const std::string& name, T* value)
	{
		watch_param<T>(name, value);
	}

	/** getter for number of parameters
	 * @return number of parameters
	 */
	virtual int32_t get_num_parameters()
	{
		ParametersMap parameters = filter(ParameterProperties::HYPER);
		return (int32_t)parameters.size();

		//return m_params.get_num_elements();
	}

	/** Adds all parameters from another instance to this one
	 *
	 * @param params another Parameter instance
	 *
	 */
	void add_parameters(CModelSelectionParameter* params);

private:
	/** initializer */
	void init();

protected:

};
}
#endif /* __MODELSELECTIONPARAMETER_H_ */
