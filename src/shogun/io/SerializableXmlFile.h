/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __SERIALIZABLE_XML_FILE_H__
#define __SERIALIZABLE_XML_FILE_H__

#include <shogun/lib/config.h>

#ifdef HAVE_XML

#include <shogun/io/SerializableFile.h>
#include <shogun/base/DynArray.h>

#include <libxml/parser.h>
#include <libxml/tree.h>

#define STR_TRUE                   "true"
#define STR_FALSE                  "false"

#define STR_ITEM                   "i"
#define STR_STRING                 "s"
#define STR_SPARSE                 "r"

#define STR_PROP_TYPE              "type"
#define STR_PROP_IS_NULL           "is_null"
#define STR_PROP_INSTANCE_NAME     "instance_name"
#define STR_PROP_GENERIC_NAME      "generic_name"
#define STR_PROP_FEATINDEX         "feat_index"

namespace shogun
{
class CSerializableXmlFile
	:public CSerializableFile
{
	friend class SerializableXmlReader00;

	DynArray<xmlNode*> m_stack_stream;
	xmlDocPtr m_doc;
	bool m_format;

	void init(bool format);
	bool push_node(const xmlChar* name);
	bool join_node(const xmlChar* name);
	bool next_node(const xmlChar* name);
	void pop_node();

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
	explicit CSerializableXmlFile();

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param format indent output; for human readable file
	 */
	explicit CSerializableXmlFile(const char* fname, char rw='r',
								  bool format=false);

	/** default destructor */
	virtual ~CSerializableXmlFile();

	/** @return object name */
	virtual const char* get_name() const {
		return "SerializableXmlFile";
	}

	virtual void close();
	virtual bool is_opened();
};
}
#endif /* HAVE_XML  */
#endif /* __SERIALIZABLE_XML_FILE_H__  */
