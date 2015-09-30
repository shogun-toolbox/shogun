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

#include <shogun/lib/config.h>

#ifdef HAVE_HDF5

#include <hdf5.h>

#include <shogun/io/SerializableFile.h>
#include <shogun/base/DynArray.h>

#define TYPE_INDEX                 H5T_NATIVE_INT32

#define STR_IS_SGSERIALIZABLE      "is_sgserializable"
#define STR_IS_SPARSE              "is_sparse"
#define STR_IS_CONT                "is_container"
#define STR_IS_NULL                "is_null"
#define STR_INSTANCE_NAME          "instance_name"
#define STR_GENERIC_NAME           "generic_name"
#define STR_CTYPE_NAME             "container_type"
#define STR_LENGTH_X               "length_x"
#define STR_LENGTH_Y               "length_y"

#define STR_GROUP_PREFIX           "$"

#define STR_SPARSE_FPTR            "features_ptr"
#define STR_SPARSEENTRY_FINDEX     "feat_index"
#define STR_SPARSEENTRY_ENTRY      "entry"

namespace shogun
{
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CSerializableHdf5File
	:public CSerializableFile
{
	friend class SerializableHdf5Reader00;

	struct type_item_t {
		explicit type_item_t(const char* name_);
		~type_item_t();

		int rank;
		hsize_t dims[2];
		hid_t dspace, dtype, dset;
		hvl_t* vltype;
		index_t y, x, sub_y;
		SGSparseVectorEntry<char>* sparse_ptr;
		const char* name;
	};

	DynArray<type_item_t*> m_stack_type;
	DynArray<hid_t> m_stack_h5stream;

	static hid_t sizeof_sparsetype();
	static hid_t new_sparsetype();
	static hobj_ref_t* get_ref_sparstype(void* sparse_buf);
	static hid_t new_sparseentrytype(EPrimitiveType ptype);
	static hid_t ptype2hdf5(EPrimitiveType ptype);
	static hid_t new_stype2hdf5(EStructType stype,
								EPrimitiveType ptype);
	static bool isequal_stype2hdf5(EStructType stype,
								   EPrimitiveType ptype, hid_t htype);
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

	bool group_create(const char* name, const char* prefix);
	bool group_open(const char* name, const char* prefix);
	bool group_close();

protected:
	virtual TSerializableReader* new_reader(
		char* dest_version, size_t n);

	virtual bool write_scalar_wrapped(
		const TSGDataType* type, const void* param);

	virtual bool write_cont_begin_wrapped(
		const TSGDataType* type, index_t len_real_y,
		index_t len_real_x);
	virtual bool write_cont_end_wrapped(
		const TSGDataType* type, index_t len_real_y,
		index_t len_real_x);

	virtual bool write_string_begin_wrapped(
		const TSGDataType* type, index_t length);
	virtual bool write_string_end_wrapped(
		const TSGDataType* type, index_t length);

	virtual bool write_stringentry_begin_wrapped(
		const TSGDataType* type, index_t y);
	virtual bool write_stringentry_end_wrapped(
		const TSGDataType* type, index_t y);

	virtual bool write_sparse_begin_wrapped(
		const TSGDataType* type, index_t length);
	virtual bool write_sparse_end_wrapped(
		const TSGDataType* type, index_t length);

	virtual bool write_sparseentry_begin_wrapped(
		const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
		index_t feat_index, index_t y);
	virtual bool write_sparseentry_end_wrapped(
		const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
		index_t feat_index, index_t y);

	virtual bool write_item_begin_wrapped(
		const TSGDataType* type, index_t y, index_t x);
	virtual bool write_item_end_wrapped(
		const TSGDataType* type, index_t y, index_t x);

	virtual bool write_sgserializable_begin_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitiveType generic);
	virtual bool write_sgserializable_end_wrapped(
		const TSGDataType* type, const char* sgserializable_name,
		EPrimitiveType generic);

	virtual bool write_type_begin_wrapped(
		const TSGDataType* type, const char* name,
		const char* prefix);
	virtual bool write_type_end_wrapped(
		const TSGDataType* type, const char* name,
		const char* prefix);

public:
	/** default constructor */
	explicit CSerializableHdf5File();

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 */
	explicit CSerializableHdf5File(const char* fname, char rw='r');

	/** default destructor */
	virtual ~CSerializableHdf5File();

	/** @return object name */
	virtual const char* get_name() const {
		return "SerializableHdf5File";
	}

	virtual void close();
	virtual bool is_opened();
};
}
#endif /* HAVE_HDF5  */
#endif /* __SERIALIZABLE_HDF5_FILE_H__  */
