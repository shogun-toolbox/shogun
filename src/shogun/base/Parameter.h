/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Written (W) 2011-2013 Heiko Strathmann
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>
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

	/** Allocates data for this instance from scratch. This is one of the core
	 * methods in parameter migration. It is used if parameters have to be
	 * loaded from file without having a class instance to put the data into.
	 * Namely, the data length variables are allocated,
	 * for numeric scalars, the memory is allocated,
	 * for SG_OBJECT scalars, a pointer to an CSGObject is allocated,
	 * for non-scalars, the pointer to the data is allocated
	 * for non-scalars, the actual data is also allocated via cont_new()
	 *
	 * @param dims desired length of the data
	 * @param new_cont_call whether new_cont should be called, if false, only scalar
	 * non-sgobject data will be allocated (needed for migration)
	 * */
	void allocate_data_from_scratch(SGVector<index_t> dims,	bool new_cont_call=true);

	/** Given another TParameter instance (with same type, except for lengths)
	 * all its data is copied to the current one. This means in case of numeric
	 * scalars that the value is copied and in SG_OBJECT scalars and any arrays
	 * that the pointer to the data is copied. The old data is overwritten.
	 * Old SG_OBJECTS are SG_UNREF'ed and the new ones are SG_REF'ed.
	 * @param source source TParameter instance to copy from */
	void copy_data(const TParameter* source);

	/** Numerically this instance with another instance. Compares recursively
	 * in case of non-numerical parameters
	 *
	 * @param other other instance to compare with
	 * @param accuracy accuracy for numerical comparison
	 * @return true if given parameter instance is equal, false otherwise
	 */
	bool equals(TParameter* other, float64_t accuracy=0.0);

	/** Given two pointers to a scalar element of a given primitive-type, this
	 * method compares the values up to a given accuracy.
	 *
	 * If the type of the data is SGObject, recursively calls equals on the
	 * object.
	 *
	 * @param ptype primitive type of both data
	 * @param data1 pointer 1
	 * @param data2 pointer 2
	 * @param accuracy accuracy to compare
	 * @return whether the data was equal
	 */
	static bool compare_ptype(EPrimitiveType ptype, void* data1, void* data2,
			floatmax_t accuracy=0.0);

	/** Given two pointers to a string element of a given primitive-type, this
	 * method compares the values up to a given accuracy.
	 *
	 * If the type of the data is SGObject, recursively calls equals on the
	 * object.
	 *
	 * @param stype string type of both data
	 * @param ptype primitive type of both data
	 * @param data1 pointer 1
	 * @param data2 pointer 2
	 * @param accuracy accuracy to compare
	 * @return whether the data was equal
	 */
	static bool compare_stype(EStructType stype, EPrimitiveType ptype,
			void* data1, void* data2, floatmax_t accuracy=0.0);

	/** copy primitive type from source to target
	 *
	 * @param ptype the primitive type
	 * @param source from where to copy
	 * @param target where to copy to
	 */
	static bool copy_ptype(EPrimitiveType ptype, void* source, void* target);

	/** copy structured type from source to target
	 *
	 * @param stype the structured type
	 * @param ptype the primitive type that the structured objects use
	 * @param source from where to copy
	 * @param target where to copy to
	 */
	static bool copy_stype(EStructType stype, EPrimitiveType ptype,
				void* source, void* target);

	/** copy this to parameter target
	 *
	 * @param target where this should be copied to
	 */
	bool copy(TParameter* target);



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

	/** @return true if data was not allocated by a class which registered
	 * its parameter, but from scratch using allocate_data_from_scratch */
	bool m_was_allocated_from_scratch;

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

	/* load from serializable file
	 * @param file source file
	 * @param prefix prefix
	virtual bool load(CSerializableFile* file, const char* prefix="");
	 */

	/** getter for number of parameters
	 * @return number of parameters
	 */
	virtual int32_t get_num_parameters()
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
	void add(SGSparseMatrix<CSGObject*>* param,
					const char* name, const char* description="");
protected:

	/** array of parameters */
	DynArray<TParameter*> m_params(4);

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
