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

	void print(CIO* io, const char* prefix);
	bool save(CSerializableFile* file, const char* prefix="");
	bool load(CSerializableFile* file, const char* prefix="");

	TSGDataType m_datatype;
	void* m_parameter;
	char* m_name;
	char* m_description;

private:
	char* new_prefix(const char* s1, const char* s2);
	bool save_scalar(CSerializableFile* file, const void* param,
					 const char* prefix);
	bool load_scalar(CSerializableFile* file, void* param,
					 const char* prefix);
};

/* Must not be an CSGObject to prevent a recursive call of
 * constructors.
 */
class CParameter
{
	CIO* io;

protected:
	DynArray<TParameter*> m_params;

	virtual void add_type(const TSGDataType* type, void* param,
						  const char* name,
						  const char* description);

public:
	explicit CParameter(CIO* io_);
	virtual ~CParameter(void);

	virtual void print(const char* prefix="");
	virtual bool save(CSerializableFile* file, const char* prefix="");
	virtual bool load(CSerializableFile* file, const char* prefix="");

	inline virtual int32_t get_num_parameters(void)
	{
		return m_params.get_num_elements();
	}

	template<class T> void add(
		T* param, const char* name, const char* description="");
	template<class T> void add_vector(
		T** param, index_t* length, const char* name,
		const char* description="");
	template<class T> void add_matrix(
		T** param, index_t* length_y, index_t* length_x,
		const char* name, const char* description="");

};
}
#endif //__PARAMETER_H__
