/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __SERIALIZABLE_JSON_FILE_H__
#define __SERIALIZABLE_JSON_FILE_H__

#include <lib/config.h>
#ifdef HAVE_JSON

#include <json.h>

#include <io/SerializableFile.h>
#include <base/DynArray.h>

#define STR_KEY_TYPE               "type"
#define STR_KEY_DATA               "data"
#define STR_KEY_INSTANCE_NAME      "instance_name"
#define STR_KEY_INSTANCE           "instance"
#define STR_KEY_GENERIC_NAME       "generic_name"
#define STR_KEY_SPARSE_FEATURES    "features"
#define STR_KEY_SPARSE_FEATINDEX   "feat_index"
#define STR_KEY_SPARSE_ENTRY       "entry"

namespace shogun
{
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CSerializableJsonFile
	:public CSerializableFile
{
	friend class SerializableJsonReader00;

	DynArray<json_object*> m_stack_stream;

	void init(const char* fname);
	void push_object(json_object* o);
	void pop_object();

	static bool get_object_any(json_object** dest, json_object* src,
							   const char* key);
	static bool get_object(json_object** dest, json_object* src,
						   const char* key, json_type t);

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
	explicit CSerializableJsonFile();

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 */
	explicit CSerializableJsonFile(const char* fname, char rw='r');

	/** default destructor */
	virtual ~CSerializableJsonFile();

	/** @return object name */
	virtual const char* get_name() const {
		return "SerializableJsonFile";
	}

	virtual void close();
	virtual bool is_opened();
};
}
#endif /* HAVE_JSON  */
#endif /* __SERIALIZABLE_JSON_FILE_H__  */
