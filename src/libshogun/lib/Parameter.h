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
	~TParameter(void);

	void print(IO* io, const char* prefix);
	bool save(IO* io, CSerializableFile* file, const char* prefix="");
	bool load(IO* io, CSerializableFile* file, const char* prefix="");

	TSGDataType m_datatype;
	void* m_parameter;
	char* m_name;
	char* m_description;

private:
	char* new_prefix(const char* s1, const char* s2);
	void new_cont(index_t new_len_y, index_t new_len_x);
	bool new_sgserial(IO* io, CSGSerializable** param,
					  EPrimitveType generic,
					  const char* sgserializable_name,
					  const char* prefix);
	bool save_scalar(IO* io, CSerializableFile* file,
					 const void* param, const char* prefix);
	bool load_scalar(IO* io, CSerializableFile* file,
					 void* param, const char* prefix);
};

/* Must not be an CSGObject to prevent a recursive call of
 * constructors.
 */
class Parameter
{
protected:
	DynArray<TParameter*> m_params;

	virtual void add_type(const TSGDataType* type, void* param,
						  const char* name,
						  const char* description);

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
	void add(CSGSerializable** param,
			 const char* name, const char* description="");

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
	void add_vector(CSGSerializable*** param, index_t* length,
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
	void add_matrix(CSGSerializable*** param,
					index_t* length_y, index_t* length_x,
					const char* name, const char* description="");
};
}
#endif //__PARAMETER_H__
