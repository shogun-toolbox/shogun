/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/DataType.h>
#include <shogun/io/SerializableFile.h>
#include <shogun/base/DynArray.h>

namespace shogun
{
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

	/** print with prefix
	 * @param prefix prefix to print
	 */
	void print(const char* prefix);

	/** save to serializable file
	 * @param file destination file
	 * @param prefix prefix
	 */
	bool save(CSerializableFile* file, const char* prefix="");

	/** load from serializable file
	 * @param file source file
	 * @param prefix prefix
	 */
	bool load(CSerializableFile* file, const char* prefix="");

	/** TODO documentation */
	void allocate_data_from_scratch(index_t len_y, index_t len_x);

	/** TODO documentation */
	void copy_data(const TParameter* source);

	/** Frees everything of this TParameter except for the data.
	 * Namely, length variables of type, data pointer for non-scalars of
	 * PT_SGOBJECT scalars.
	 * If container type is CT_SCALAR and numeric, the data is also deleted.
	 * Do not call unless this TParameter instance was created by
	 * allocate_data_from scratch because some of these variables may lie on
	 * stack if not. This method is used in parameter version migration
	 */
	void delete_all_but_data();

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

	/** if this is set true, the data, m_parameter points to, m_parameter
	 * itself, and possible lengths of the type will be deleted in destructor.
	 * This is needed because in data migration, TParameter instances are
	 * created from scratch without having a class instance and allocated data
	 * has to ne deleted in this case.
	 * The only way to set this is via an alternate constructor, false by
	 * default */
	bool m_delete_data;

private:
	char* new_prefix(const char* s1, const char* s2);
	void delete_cont();
	void new_cont(index_t new_len_y, index_t new_len_x);
	bool new_sgserial(CSGObject** param, EPrimitiveType generic,
					  const char* sgserializable_name,
					  const char* prefix);
	bool save_ptype(CSerializableFile* file, const void* param,
					const char* prefix);
	bool load_ptype(CSerializableFile* file, void* param,
					const char* prefix);
	bool save_stype(CSerializableFile* file, const void* param,
					const char* prefix);
	bool load_stype(CSerializableFile* file, void* param,
					const char* prefix);
};

/** @brief Parameter class 
 * 
 * Must not be an CSGObject to prevent a recursive call of
 * constructors.
 */
class Parameter
{
public:
	/** explicit constructor */
	explicit Parameter();
	/** destructor */
	virtual ~Parameter();

	/** print
	 * @param prefix prefix
	 */
	virtual void print(const char* prefix="");

	/** save to serializable file
	 * @param file destination file
	 * @param prefix prefix
	 */
	virtual bool save(CSerializableFile* file, const char* prefix="");

	/** load from serializable file
	 * @param file source file
	 * @param prefix prefix
	 */
	virtual bool load(CSerializableFile* file, const char* prefix="");

	/** getter for number of parameters
	 * @return number of parameters
	 */
	inline virtual int32_t get_num_parameters()
	{
		return m_params.get_num_elements();
	}

	/** Takes another Parameter instance and sets all parameters of this
	 * instance (with equal name) to the values of the provided one.
	 * (Note that if CSGObjects are replaced, the old ones are SG_UNREFed
	 * and the new ones are SG_REFed)
	 * Currently only works for any float64_t and CSGObject type.
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
	void add(CSGObject** param,
			 const char* name, const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<bool>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<char>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<int8_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<uint8_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<int16_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<uint16_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<int32_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<uint32_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<int64_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<uint64_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<float32_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<float64_t>* param, const char* name,
			 const char* description="");
	/** add param
	 * @param param parameter itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGString<floatmax_t>* param, const char* name,
			 const char* description="");
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
	void add_vector(CSGObject*** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<bool>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<char>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<int8_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<uint8_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<int16_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<uint16_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<int32_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<uint32_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<int64_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<uint64_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<float32_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<float64_t>** param, index_t* length,
					const char* name, const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param length length of vector
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_vector(SGString<floatmax_t>** param, index_t* length,
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
	void add(SGVector<CSGObject*>* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<bool> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<char> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<int8_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<uint8_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<int16_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<uint16_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<int32_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<uint32_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<int64_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<uint64_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<float32_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<float64_t> >* param, const char* name,
					const char* description="");
	/** add vector param
	 * @param param parameter vector itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGVector<SGString<floatmax_t> >* param, const char* name,
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
	void add_matrix(CSGObject*** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<bool>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<char>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<int8_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<uint8_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<int16_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<uint16_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<int32_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<uint32_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<int64_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<uint64_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<float32_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<float64_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param length_y y size of matrix
	 * @param length_x x size of matrix
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add_matrix(SGString<floatmax_t>** param,
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
	void add(SGMatrix<CSGObject*>* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<bool> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<char> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<int8_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<uint8_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<int16_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<uint16_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<int32_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<uint32_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<int64_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<uint64_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<float32_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<float64_t> >* param, const char* name,
					const char* description="");
	/** add matrix param
	 * @param param parameter matrix itself
	 * @param name name of parameter
	 * @param description description of parameter
	 */
	void add(SGMatrix<SGString<floatmax_t> >* param, const char* name,
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
