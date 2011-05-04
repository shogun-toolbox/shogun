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

#include "lib/common.h"
#include "lib/io.h"
#include "lib/DataType.h"
#include "lib/SerializableFile.h"
#include "base/DynArray.h"

namespace shogun
{
struct TParameter
{
	explicit TParameter(const TSGDataType* datatype, void* parameter,
						const char* name, const char* description);
	~TParameter();

	void print(const char* prefix);
	bool save(CSerializableFile* file, const char* prefix="");
	bool load(CSerializableFile* file, const char* prefix="");

	TSGDataType m_datatype;
	void* m_parameter;
	char* m_name;
	char* m_description;

private:
	char* new_prefix(const char* s1, const char* s2);
	void delete_cont(void);
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

/* Must not be an CSGObject to prevent a recursive call of
 * constructors.
 */
class Parameter
{
public:
	explicit Parameter(void);
	virtual ~Parameter(void);

	virtual void print(const char* prefix="");
	virtual bool save(CSerializableFile* file, const char* prefix="");
	virtual bool load(CSerializableFile* file, const char* prefix="");

	inline virtual int32_t get_num_parameters(void)
	{
		return m_params.get_num_elements();
	}

	inline TParameter* get_parameter(int32_t idx) { return m_params.get_element(idx); }

	/** Takes another Parameter instance and sets all parameters of this
	 * instance (with equal name) to the values of the provided one.
	 * Currently only works for float64_t and CSGObject types.
	 *
	 * @param params another Parameter instance
	 */
	void set_from_parameters(Parameter* params);

	/* ************************************************************ */
	/* Scalar wrappers  */

	void add(bool* param, const char* name,
			 const char* description="");
	void add(char* param, const char* name,
			 const char* description="");
	void add(int8_t* param, const char* name,
			 const char* description="");
	void add(uint8_t* param, const char* name,
			 const char* description="");
	void add(int16_t* param, const char* name,
			 const char* description="");
	void add(uint16_t* param, const char* name,
			 const char* description="");
	void add(int32_t* param, const char* name,
			 const char* description="");
	void add(uint32_t* param, const char* name,
			 const char* description="");
	void add(int64_t* param, const char* name,
			 const char* description="");
	void add(uint64_t* param, const char* name,
			 const char* description="");
	void add(float32_t* param, const char* name,
			 const char* description="");
	void add(float64_t* param, const char* name,
			 const char* description="");
	void add(floatmax_t* param, const char* name,
			 const char* description="");

	void add(CSGObject** param,
			 const char* name, const char* description="");

	void add(TString<bool>* param, const char* name,
			 const char* description="");
	void add(TString<char>* param, const char* name,
			 const char* description="");
	void add(TString<int8_t>* param, const char* name,
			 const char* description="");
	void add(TString<uint8_t>* param, const char* name,
			 const char* description="");
	void add(TString<int16_t>* param, const char* name,
			 const char* description="");
	void add(TString<uint16_t>* param, const char* name,
			 const char* description="");
	void add(TString<int32_t>* param, const char* name,
			 const char* description="");
	void add(TString<uint32_t>* param, const char* name,
			 const char* description="");
	void add(TString<int64_t>* param, const char* name,
			 const char* description="");
	void add(TString<uint64_t>* param, const char* name,
			 const char* description="");
	void add(TString<float32_t>* param, const char* name,
			 const char* description="");
	void add(TString<float64_t>* param, const char* name,
			 const char* description="");
	void add(TString<floatmax_t>* param, const char* name,
			 const char* description="");

	void add(TSparse<bool>* param, const char* name,
			 const char* description="");
	void add(TSparse<char>* param, const char* name,
			 const char* description="");
	void add(TSparse<int8_t>* param, const char* name,
			 const char* description="");
	void add(TSparse<uint8_t>* param, const char* name,
			 const char* description="");
	void add(TSparse<int16_t>* param, const char* name,
			 const char* description="");
	void add(TSparse<uint16_t>* param, const char* name,
			 const char* description="");
	void add(TSparse<int32_t>* param, const char* name,
			 const char* description="");
	void add(TSparse<uint32_t>* param, const char* name,
			 const char* description="");
	void add(TSparse<int64_t>* param, const char* name,
			 const char* description="");
	void add(TSparse<uint64_t>* param, const char* name,
			 const char* description="");
	void add(TSparse<float32_t>* param, const char* name,
			 const char* description="");
	void add(TSparse<float64_t>* param, const char* name,
			 const char* description="");
	void add(TSparse<floatmax_t>* param, const char* name,
			 const char* description="");

	/* ************************************************************ */
	/* Vector wrappers  */

	void add_vector(bool** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(char** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(int8_t** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(uint8_t** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(int16_t** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(uint16_t** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(int32_t** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(uint32_t** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(int64_t** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(uint64_t** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(float32_t** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(float64_t** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(floatmax_t** param, index_t* length,
					const char* name, const char* description="");

	void add_vector(CSGObject*** param, index_t* length,
					const char* name, const char* description="");

	void add_vector(TString<bool>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TString<char>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TString<int8_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TString<uint8_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TString<int16_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TString<uint16_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TString<int32_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TString<uint32_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TString<int64_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TString<uint64_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TString<float32_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TString<float64_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TString<floatmax_t>** param, index_t* length,
					const char* name, const char* description="");

	void add_vector(TSparse<bool>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TSparse<char>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TSparse<int8_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TSparse<uint8_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TSparse<int16_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TSparse<uint16_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TSparse<int32_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TSparse<uint32_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TSparse<int64_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TSparse<uint64_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TSparse<float32_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TSparse<float64_t>** param, index_t* length,
					const char* name, const char* description="");
	void add_vector(TSparse<floatmax_t>** param, index_t* length,
					const char* name, const char* description="");

	/* ************************************************************ */
	/* Matrix wrappers  */

	void add_matrix(bool** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(char** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(int8_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(uint8_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(int16_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(uint16_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(int32_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(uint32_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(int64_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(uint64_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(float32_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(float64_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(floatmax_t** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");

	void add_matrix(CSGObject*** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");

	void add_matrix(TString<bool>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TString<char>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TString<int8_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TString<uint8_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TString<int16_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TString<uint16_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TString<int32_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TString<uint32_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TString<int64_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TString<uint64_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TString<float32_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TString<float64_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TString<floatmax_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");

	void add_matrix(TSparse<bool>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TSparse<char>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TSparse<int8_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TSparse<uint8_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TSparse<int16_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TSparse<uint16_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TSparse<int32_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TSparse<uint32_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TSparse<int64_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TSparse<uint64_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TSparse<float32_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TSparse<float64_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
	void add_matrix(TSparse<floatmax_t>** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");

protected:
	DynArray<TParameter*> m_params;

	virtual void add_type(const TSGDataType* type, void* param,
						  const char* name,
						  const char* description);

};
}
#endif //__PARAMETER_H__
