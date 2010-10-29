/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __SERIALIZABLE_HDF5_FILE_H__
#define __SERIALIZABLE_HDF5_FILE_H__

#include "lib/config.h"
#ifdef HAVE_HDF5

#include <hdf5.h>

#include "lib/SerializableFile.h"
#include "lib/DynamicArray.h"

namespace shogun
{
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CSerializableHDF5File
	:public CSerializableFile
{
	struct type_item_t {
		explicit type_item_t(void);
		~type_item_t(void);

		int rank;
		hsize_t dims[2];
		hid_t dspace, dtype, dset;
	};

	CDynamicArray<type_item_t*> m_stack_type;
	CDynamicArray<hid_t> m_stack_h5stream;

	static hid_t new_ptype2hdf5(EPrimitveType ptype);
	static bool isequal_ptype2hdf5(EPrimitveType ptype, hid_t htype);
	static bool index2string(char* dest, size_t n, EContainerType ctype,
							 index_t y, index_t x);

	void init(const char* fname);
	bool dspace_select(EContainerType ctype, index_t y, index_t x);

	bool attr_write_scalar(hid_t datatype, const char* name,
						   const void* val);
	bool attr_write_string(const char* name, const char* val);
	bool attr_exists(const char* name);
	size_t attr_get_size(const char* name);
	bool attr_read_scalar(hid_t datatype, const char* name, void* val);
	bool attr_read_string(const char* name, char* val, size_t n);

	bool group_create(const char* name);
	bool group_open(const char* name);
	bool group_close(void);

protected:
	virtual bool write_scalar_wrapped(const TSGDataType* type,
									  const void* param);
	virtual bool read_scalar_wrapped(const TSGDataType* type,
									 void* param);

	virtual bool write_cont_begin_wrapped(const TSGDataType* type,
										  index_t len_real_y,
										  index_t len_real_x);
	virtual bool read_cont_begin_wrapped(const TSGDataType* type,
										 index_t* len_read_y,
										 index_t* len_read_x);

	virtual bool write_cont_end_wrapped(const TSGDataType* type,
										index_t len_real_y,
										index_t len_real_x);
	virtual bool read_cont_end_wrapped(const TSGDataType* type,
									   index_t len_read_y,
									   index_t len_read_x);

	virtual bool write_item_begin_wrapped(const TSGDataType* type,
										  index_t y, index_t x);
	virtual bool read_item_begin_wrapped(const TSGDataType* type,
										 index_t y, index_t x);

	virtual bool write_item_end_wrapped(const TSGDataType* type,
										index_t y, index_t x);
	virtual bool read_item_end_wrapped(const TSGDataType* type,
									   index_t y, index_t x);

	virtual bool write_sgserializable_begin_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitveType generic);
	virtual bool read_sgserializable_begin_wrapped(
		const TSGDataType* type, char* sgserializable_name,
		EPrimitveType* generic);

	virtual bool write_sgserializable_end_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitveType generic);
	virtual bool read_sgserializable_end_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitveType generic);

	virtual bool write_type_begin_wrapped(const TSGDataType* type,
										  const char* name,
										  const char* prefix);
	virtual bool read_type_begin_wrapped(const TSGDataType* type,
										 const char* name,
										 const char* prefix);

	virtual bool write_type_end_wrapped(const TSGDataType* type,
										const char* name,
										const char* prefix);
	virtual bool read_type_end_wrapped(const TSGDataType* type,
									   const char* name,
									   const char* prefix);

public:
	/** default constructor */
	explicit CSerializableHDF5File(void);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 */
	explicit CSerializableHDF5File(const char* fname, char rw='r');

	/** default destructor */
	virtual ~CSerializableHDF5File();

	/** @return object name */
	inline virtual const char* get_name() const {
		return "SerializableHDF5File";
	}

	virtual void close(void);
	virtual bool is_opened(void);
};
}
#endif /* HAVE_HDF5  */
#endif /* __SERIALIZABLE_HDF5_FILE_H__  */
