/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn, Jacob Walker,
 *          Thoralf Klein, Soumyajit De, Yuyu Zhang, Evan Shelhamer
 */
#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <shogun/base/DynArray.h>

namespace shogun
{

class SGObject;
template <class T> class SGMatrix;
template <class T> class SGSparseMatrix;
template <class T> class SGVector;
template <class T> class SGSparseVector;

/** @brief parameter struct */
struct TParameter
{
	/** explicit constructor
	 * @param datatype datatype
	 * @param parameter pointer to parameter
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	explicit TParameter(const TSGDataType* datatype, void* parameter,
						const char* name, const char* description);

	/** destructor */
	~TParameter();

	/** operator for comparison, (by string m_name) */
	bool operator==(const TParameter& other) const;

	/** operator for comparison (by string m_name) */
	bool operator<(const TParameter& other) const;

	/** operator for comparison (by string m_name) */
	bool operator>(const TParameter& other) const;

	/** type of parameter */
	TSGDataType m_datatype;
	/** pointer to parameter */
	void* m_parameter;
	/** name of parameter */
	char* m_name;
	/** description of parameter */
	char* m_description;

	/** Incrementally get a hash from parameter value
	 *
	 * @param hash current hash value
	 * @param carry value for incremental murmur hashing
	 * @param total_length byte length of parameters. Function will
	 * add byte length to received value
	 *
	 */
	void get_incremental_hash(
			uint32_t& hash, uint32_t& carry, uint32_t& total_length);

	/** test if parameter can be validly accessed, e.g., in case of a
	 * list/vector/matrix of objects the list/vector/matrix has non-zero length
	 */
	bool is_valid();

private:
	char* new_prefix(const char* s1, const char* s2);
	void delete_cont();
	void new_cont(SGVector<index_t> dims);
	bool new_sgserial(SGObject** param, EPrimitiveType generic,
					  const char* sgserializable_name,
					  const char* prefix);
};

/** @brief Parameter class
 *
 * Must not be an SGObject to prevent a recursive call of
 * constructors.
 */
class Parameter
{
public:
	/** explicit constructor */
	explicit Parameter();
	/** destructor */
	virtual ~Parameter();

	/** getter for number of parameters
	 * @return number of parameters
	 */
	virtual int32_t get_num_parameters()
	{
		return m_params.get_num_elements();
	}

	/** Takes another Parameter instance and sets all parameters of this
	 * instance (with equal name) to the values of the provided one.
	 * (Note that if SGObjects are replaced, the old ones are SG_UNREFed
	 * and the new ones are SG_REFed)
	 * Currently only works for any float64_t and SGObject type.
	 *
	 * @param params another Parameter instance
	 */
	void set_from_parameters(Parameter* params);

	/** Adds all parameters from another instance to this one
	 *
	 * @param params another Parameter instance
	 *
	 */
	void add_parameters(Parameter* params);

	/** Checks if a parameter with the spcified name is included
	 *
	 * @return true if parameter with name is included
	 */
	bool contains_parameter(const char* name);

	/** Getter for TParameter elements (Does not to bound checking)
	 *
	 * @param idx desired index
	 * @return pointer to the TParameter with the specified index
	 */
	inline TParameter* get_parameter(int32_t idx)
	{
		return m_params.get_element(idx);
	}

	/** Getter for Tparameter elements by name
	 *
	 * @param name name of desired parameter
	 * @return parameter with desired name, NULL if non such found
	 */
	inline TParameter* get_parameter(const char* name)
	{
		TParameter* result=NULL;

		for (index_t i=0; i<m_params.get_num_elements(); ++i)
		{
			result=m_params.get_element(i);
			if (!strcmp(name, result->m_name))
				break;
			else
				result=NULL;
		}

		return result;
	}

	/* ************************************************************ */
	/* Scalar wrappers  */

	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(bool* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(char* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(int8_t* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(uint8_t* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(int16_t* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(uint16_t* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(int32_t* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(uint32_t* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(int64_t* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(uint64_t* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(float32_t* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(float64_t* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(floatmax_t* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(complex128_t* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGObject** param,
			 const char* name, const char* description="");

	template <typename T, std::enable_if_t<std::is_base_of<SGObject, T>::value,
		                                   T>* = nullptr>
	void add(T** param, const char* name, const char* description = "")
	{
		TSGDataType type(CT_SCALAR, ST_NONE, PT_SGOBJECT);
		add_type(&type, (SGObject**)param, name, description);
	}

	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<bool>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<char>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<int8_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<uint8_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<int16_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<uint16_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<int32_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<uint32_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<int64_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<uint64_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<float32_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<float64_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<floatmax_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseVector<complex128_t>* param, const char* name,
			 const char* description="");

	/* ************************************************************ */
	/* Vector wrappers  */

	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(bool** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(char** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(int8_t** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(uint8_t** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(int16_t** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(uint16_t** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(int32_t** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(uint32_t** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(int64_t** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(uint64_t** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(float32_t** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(float64_t** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(floatmax_t** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(complex128_t** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGObject*** param, index_t* length,
					const char* name, const char* description="");
		/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<bool>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<char>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<int8_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<uint8_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<int16_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<uint16_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<int32_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<uint32_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<int64_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<uint64_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<float32_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<float64_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGVector<floatmax_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<bool>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<char>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<int8_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<uint8_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<int16_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<uint16_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<int32_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<uint32_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<int64_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<uint64_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<float32_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<float64_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<floatmax_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGSparseVector<complex128_t>** param, index_t* length,
					const char* name, const char* description="");


	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<bool>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<char>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<int8_t>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<uint8_t>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<int16_t>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<uint16_t>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<int32_t>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<uint32_t>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<int64_t>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<uint64_t>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<float32_t>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<float64_t>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<floatmax_t>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<complex128_t>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGObject*>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<bool> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<char> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<int8_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<uint8_t> >* param,const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<int16_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<uint16_t> >* param,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<int32_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<uint32_t> >* param,const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<int64_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<uint64_t> >* param,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<float32_t> >* param,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<float64_t> >* param,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<floatmax_t> >* param,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGSparseVector<complex128_t> >* param,
					const char* name, const char* description="");

	/* ************************************************************ */
	/* Matrix wrappers  */

	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(bool** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(char** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(int8_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(uint8_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(int16_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(uint16_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(int32_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(uint32_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(int64_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(uint64_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(float32_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(float64_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(floatmax_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(complex128_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGObject*** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<bool>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<char>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<int8_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<uint8_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<int16_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<uint16_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<int32_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<uint32_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<int64_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<uint64_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<float32_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<float64_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<floatmax_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGSparseVector<complex128_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<bool>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<char>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<int8_t>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<uint8_t>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<int16_t>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<uint16_t>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<int32_t>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<uint32_t>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<int64_t>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<uint64_t>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<float32_t>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<float64_t>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<floatmax_t>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<complex128_t>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGObject*>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<bool> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<char> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<int8_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<uint8_t> >* param,const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<int16_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<uint16_t> >* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<int32_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<uint32_t> >* param,const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<int64_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<uint64_t> >* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<float32_t> >* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<float64_t> >* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<floatmax_t> >* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGSparseVector<complex128_t> >* param,
					const char* name, const char* description="");

	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<bool>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<char>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<int8_t>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<uint8_t>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<int16_t>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<uint16_t>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<int32_t>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<uint32_t>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<int64_t>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<uint64_t>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<float32_t>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<float64_t>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<floatmax_t>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<complex128_t>* param,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGSparseMatrix<SGObject*>* param,
					const char* name, const char* description="");
protected:

	/** array of parameters */
	DynArray<TParameter*> m_params;

	/** add new type
	 * @param type type to be added
	 * @param param pointer to parameter
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	virtual void add_type(const TSGDataType* type, void* param,
						  const char* name,
						  const char* description);
};
}
#endif //__PARAMETER_H__
