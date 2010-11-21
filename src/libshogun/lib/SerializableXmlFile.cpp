/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "lib/config.h"
#ifdef HAVE_XML

#include "lib/SerializableXmlFile.h"

#define STR_ROOT_NAME \
	"_SHOGUN_SERIALIZABLE_XML_FILE_V_00_"

#define STR_TRUE                   "true"
#define STR_FALSE                  "false"

#define STR_ITEM                   "i"
#define STR_STRING                 "s"
#define STR_SPARSE                 "r"

#define STR_PROP_TYPE              "type"
#define STR_PROP_IS_NULL           "is_null"
#define STR_PROP_INSTANCE_NAME     "instance_name"
#define STR_PROP_GENERIC_NAME      "generic_name"
#define STR_PROP_VECINDEX          "vec_index"
#define STR_PROP_FEATINDEX         "feat_index"

using namespace shogun;

CSerializableXmlFile::CSerializableXmlFile(void)
	:CSerializableFile() { init("", false); }

CSerializableXmlFile::CSerializableXmlFile(const char* fname, char rw,
										   bool format)
	:CSerializableFile()
{
	CSerializableFile::init(NULL, rw, fname);
	init(fname, format);
}

CSerializableXmlFile::~CSerializableXmlFile()
{
	close();
}

bool
CSerializableXmlFile::push_node(const xmlChar* name)
{
	xmlNode* node
		= xmlNewChild(m_stack_stream.back(), NULL, name, NULL);

	m_stack_stream.push_back(node);

	return node != NULL;
}

bool
CSerializableXmlFile::join_node(const xmlChar* name)
{
	for (xmlNode* cur=m_stack_stream.back()->children; cur!=NULL;
		 cur=cur->next) {
		if (cur->type != XML_ELEMENT_NODE
			|| xmlStrcmp(cur->name, name) != 0) continue;

		m_stack_stream.push_back(cur);
		return true;
	}

	return false;
}

bool
CSerializableXmlFile::next_node(const xmlChar* name)
{
	for (xmlNode* cur=m_stack_stream.back()->next; cur!=NULL;
		 cur=cur->next) {
		if (cur->type != XML_ELEMENT_NODE
			|| xmlStrcmp(cur->name, name) != 0) continue;

		pop_node();
		m_stack_stream.push_back(cur);
		return true;
	}

	return false;
}

void
CSerializableXmlFile::pop_node(void)
{
	m_stack_stream.pop_back();
}

void
CSerializableXmlFile::init(const char* fname, bool format)
{
	m_format = format;

	LIBXML_TEST_VERSION;

	if (m_filename == NULL || *m_filename == '\0') {
		SG_WARNING("Filename not given for opening file!\n");
		close(); return;
	}

	xmlNode* tmp; xmlChar* name;
	switch (m_task) {
	case 'r':
		if ((m_doc = xmlReadFile(fname, NULL, 0)) == NULL
			|| (tmp = xmlDocGetRootElement(m_doc)) == NULL) {
			SG_WARNING("Could not open file `%s' for reading!\n",
					   fname);
			close(); return;
		}
		m_stack_stream.push_back(tmp);

		if ((name = xmlGetNodePath(m_stack_stream.back())) == NULL
			|| xmlStrcmp(BAD_CAST STR_ROOT_NAME, name+1) != 0) {
			SG_WARNING("%s: Not a Serializable XML file!\n", fname);
			close(); return;
		}
		break;
	case 'w':
		m_doc = xmlNewDoc(BAD_CAST "1.0");
		m_stack_stream.push_back(xmlNewNode(NULL,
											BAD_CAST STR_ROOT_NAME));
		xmlDocSetRootElement(m_doc, m_stack_stream.back());
		break;
	default:
		SG_WARNING("Could not open file `%s', unknown mode!\n",
				   m_filename);
		close(); return;
	}
}

void
CSerializableXmlFile::close(void)
{
	while (m_stack_stream.get_num_elements() > 0) pop_node();

	if (is_opened()) {
		if (m_task == 'w'
			&&  xmlSaveFormatFileEnc(m_filename, m_doc, "UTF-8",
									 m_format) < 0) {
			SG_WARNING("Could not close file `%s' for writing!\n",
					   m_filename);
		}

		xmlFreeDoc(m_doc); m_doc = NULL;
		xmlCleanupParser();
	}
}

bool
CSerializableXmlFile::is_opened(void)
{
	return m_doc != NULL;
}

bool
CSerializableXmlFile::write_scalar_wrapped(
	const TSGDataType* type, const void* param)
{
	string_t buf;

	switch (type->m_ptype) {
	case PT_BOOL:
		if (snprintf(buf, STRING_LEN, "%s", *(bool*) param? STR_TRUE
					 : STR_FALSE) <= 0) return false;
		break;
	case PT_CHAR:
		if (snprintf(buf, STRING_LEN, "%c", *(char*) param
				) <= 0) return false;
		break;
	case PT_INT8:
		if (snprintf(buf, STRING_LEN, "%"PRIi8, *(int8_t*) param
				) <= 0) return false;
		break;
	case PT_UINT8:
		if (snprintf(buf, STRING_LEN, "%"PRIu8, *(uint8_t*) param
				) <= 0) return false;
		break;
	case PT_INT16:
		if (snprintf(buf, STRING_LEN, "%"PRIi16, *(int16_t*) param
				) <= 0) return false;
		break;
	case PT_UINT16:
		if (snprintf(buf, STRING_LEN, "%"PRIu16, *(uint16_t*) param
				) <= 0) return false;
		break;
	case PT_INT32:
		if (snprintf(buf, STRING_LEN, "%"PRIi32, *(int32_t*) param
				) <= 0) return false;
		break;
	case PT_UINT32:
		if (snprintf(buf, STRING_LEN, "%"PRIu32, *(uint32_t*) param
				) <= 0) return false;
		break;
	case PT_INT64:
		if (snprintf(buf, STRING_LEN, "%"PRIi64, *(int64_t*) param
				) <= 0) return false;
		break;
	case PT_UINT64:
		if (snprintf(buf, STRING_LEN, "%"PRIu64, *(uint64_t*) param
				) <= 0) return false;
		break;
	case PT_FLOAT32:
		if (snprintf(buf, STRING_LEN, "%+10.16e", *(float32_t*) param
				) <= 0) return false;
		break;
	case PT_FLOAT64:
		if (snprintf(buf, STRING_LEN, "%+10.16e", *(float64_t*) param
				) <= 0) return false;
		break;
	case PT_FLOATMAX:
		if (snprintf(buf, STRING_LEN, "%+10.16Le", *(floatmax_t*)
					 param) <= 0) return false;
		break;
	case PT_SGSERIALIZABLE_PTR:
		SG_ERROR("write_scalar_wrapped(): Implementation error during"
				 " writing XmlFile!");
		return false;
	}

	xmlNodeAddContent(m_stack_stream.back(), BAD_CAST buf);
	return true;
}

bool
CSerializableXmlFile::read_scalar_wrapped(
	const TSGDataType* type, void* param)
{
	bool result = true;
	xmlChar* xml_buf;
	if ((xml_buf = xmlNodeGetContent(m_stack_stream.back())) == NULL)
		return false;
	const char* buf = (const char*) xml_buf;

	switch (type->m_ptype) {
	case PT_BOOL:
		string_t bool_buf;

		if (sscanf(buf, "%"STRING_LEN_STR"s", bool_buf) != 1)
			result = false;

		if (strcmp(buf, STR_TRUE) == 0) *(bool*) param = true;
		else if (strcmp(buf, STR_FALSE) == 0) *(bool*) param = false;
		else result = false;

		break;
	case PT_CHAR:
		if (sscanf(buf, "%c", (char*) param) != 1)
			result = false;
		break;
	case PT_INT8:
		if (sscanf(buf, "%"SCNi8, (int8_t*) param) != 1)
			result = false;
		break;
	case PT_UINT8:
		if (sscanf(buf, "%"SCNu8, (uint8_t*) param) != 1)
			result = false;
		break;
	case PT_INT16:
		if (sscanf(buf, "%"SCNi16, (int16_t*) param) != 1)
			result = false;
		break;
	case PT_UINT16:
		if (sscanf(buf, "%"SCNu16, (uint16_t*) param) != 1)
			result = false;
		break;
	case PT_INT32:
		if (sscanf(buf, "%"SCNi32, (int32_t*) param) != 1)
			result = false;
		break;
	case PT_UINT32:
		if (sscanf(buf, "%"SCNu32, (uint32_t*) param) != 1)
			result = false;
		break;
	case PT_INT64:
		if (sscanf(buf, "%"SCNi64, (int64_t*) param) != 1)
			result = false;
		break;
	case PT_UINT64:
		if (sscanf(buf, "%"SCNu64, (uint64_t*) param) != 1)
			result = false;
		break;
	case PT_FLOAT32:
		if (sscanf(buf, "%e", (float32_t*) param) != 1)
			result = false;
		break;
	case PT_FLOAT64:
		if (sscanf(buf, "%le", (float64_t*) param) != 1)
			result = false;
		break;
	case PT_FLOATMAX:
		if (sscanf(buf, "%Le", (floatmax_t*) param) != 1)
			result = false;
		break;
	case PT_SGSERIALIZABLE_PTR:
		SG_ERROR("read_scalar_wrapped(): Implementation error during"
				 " reading XmlFile!");
		result = false;
	}

	xmlFree(xml_buf);
	return result;
}

bool
CSerializableXmlFile::write_cont_begin_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	return true;
}

bool
CSerializableXmlFile::read_cont_begin_wrapped(
	const TSGDataType* type, index_t* len_read_y, index_t* len_read_x)
{
	switch (type->m_ctype) {
	case CT_SCALAR: break;
	case CT_VECTOR:
		*len_read_y = xmlChildElementCount(m_stack_stream.back());
		break;
	case CT_MATRIX:
		*len_read_x = xmlChildElementCount(m_stack_stream.back());

		for (xmlNode* cur=m_stack_stream.back()->children; cur != NULL;
			 cur=cur->next) {
			if (cur->type != XML_ELEMENT_NODE) continue;

			if (*len_read_y == 0)
				*len_read_y = xmlChildElementCount(cur);

			if (*len_read_y != (index_t) xmlChildElementCount(cur))
				return false;
		}

		break;
	}

	return true;
}

bool
CSerializableXmlFile::write_cont_end_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	if (type->m_ctype == CT_MATRIX && len_real_y *len_real_x > 0)
		pop_node();

	return true;
}

bool
CSerializableXmlFile::read_cont_end_wrapped(
	const TSGDataType* type, index_t len_read_y, index_t len_read_x)
{
	if (len_read_y > 0) pop_node();

	if (type->m_ctype == CT_MATRIX && len_read_y *len_read_x > 0)
		pop_node();

	return true;
}

bool
CSerializableXmlFile::write_string_begin_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
CSerializableXmlFile::read_string_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	*length = xmlChildElementCount(m_stack_stream.back());

	return true;
}

bool
CSerializableXmlFile::write_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
CSerializableXmlFile::read_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	if (length > 0) pop_node();

	return true;
}

bool
CSerializableXmlFile::write_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	if (!push_node(BAD_CAST STR_STRING)) return false;

	return true;
}

bool
CSerializableXmlFile::read_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	if (y == 0) {
		if (!join_node(BAD_CAST STR_STRING)) return false;
		return true;
	}

	if (!next_node(BAD_CAST STR_STRING)) return false;

	return true;
}

bool
CSerializableXmlFile::write_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	pop_node();

	return true;
}

bool
CSerializableXmlFile::read_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	return true;
}

bool
CSerializableXmlFile::write_sparse_begin_wrapped(
	const TSGDataType* type, index_t vec_index,
	index_t length)
{
	string_t buf;
	snprintf(buf, STRING_LEN, "%"PRIi32, vec_index);
	if (xmlNewProp(m_stack_stream.back(), BAD_CAST STR_PROP_VECINDEX,
				   BAD_CAST buf) == NULL) return false;

	return true;
}

bool
CSerializableXmlFile::read_sparse_begin_wrapped(
	const TSGDataType* type, index_t* vec_index,
	index_t* length)
{
	bool result = true;
	xmlChar* buf;

	if ((buf = xmlGetProp(m_stack_stream.back(), BAD_CAST
						  STR_PROP_VECINDEX)) == NULL) return false;
	if (sscanf((const char*) buf, "%"PRIi32, vec_index) != 1)
		result = false;
	xmlFree(buf); if (!result) return false;

	*length = xmlChildElementCount(m_stack_stream.back());

	return true;
}

bool
CSerializableXmlFile::write_sparse_end_wrapped(
	const TSGDataType* type, index_t vec_index,
	index_t length)
{
	return true;
}

bool
CSerializableXmlFile::read_sparse_end_wrapped(
	const TSGDataType* type, index_t* vec_index,
	index_t length)
{
	if (length > 0) pop_node();

	return true;
}

bool
CSerializableXmlFile::write_sparseentry_begin_wrapped(
	const TSGDataType* type, const TSparseEntry<char>* first_entry,
	index_t feat_index, index_t y)
{
	push_node(BAD_CAST STR_SPARSE);

	string_t buf;
	snprintf(buf, STRING_LEN, "%"PRIi32, feat_index);
	if (xmlNewProp(m_stack_stream.back(), BAD_CAST STR_PROP_FEATINDEX,
				   BAD_CAST buf) == NULL) return false;
	return true;
}

bool
CSerializableXmlFile::read_sparseentry_begin_wrapped(
	const TSGDataType* type, TSparseEntry<char>* first_entry,
	index_t* feat_index, index_t y)
{
	bool result = true;
	xmlChar* buf;

	if (y == 0) {
		if (!join_node(BAD_CAST STR_SPARSE)) return false;
	} else {
		if (!next_node(BAD_CAST STR_SPARSE)) return false;
	}

	if ((buf = xmlGetProp(m_stack_stream.back(), BAD_CAST
						  STR_PROP_FEATINDEX)) == NULL) return false;
	if (sscanf((const char*) buf, "%"PRIi32, feat_index) != 1)
		result = false;
	xmlFree(buf); if (!result) return false;

	return true;
}

bool
CSerializableXmlFile::write_sparseentry_end_wrapped(
	const TSGDataType* type, const TSparseEntry<char>* first_entry,
	index_t feat_index, index_t y)
{
	pop_node();

	return true;
}

bool
CSerializableXmlFile::read_sparseentry_end_wrapped(
	const TSGDataType* type, TSparseEntry<char>* first_entry,
	index_t* feat_index, index_t y)
{
	return true;
}

bool
CSerializableXmlFile::write_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (type->m_ctype == CT_MATRIX && y == 0) {
		if (x != 0) pop_node();

		string_t buf_x; snprintf(buf_x, STRING_LEN, "x%"PRIi32, x);
		if (!push_node(BAD_CAST buf_x)) return false;
	}

	push_node(BAD_CAST STR_ITEM);

	return true;
}

bool
CSerializableXmlFile::read_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	switch (type->m_ctype) {
	case CT_SCALAR: break;
	case CT_VECTOR:
		if (y == 0) {
			if (!join_node(BAD_CAST STR_ITEM)) return false;
			return true;
		}
		break;
	case CT_MATRIX:
		if (type->m_ctype == CT_MATRIX && y == 0) {
			if (x != 0) { pop_node(); pop_node(); }

			string_t buf_x; snprintf(buf_x, STRING_LEN, "x%"PRIi32, x);
			if (!join_node(BAD_CAST buf_x)) return false;
			if (!join_node(BAD_CAST STR_ITEM)) return false;
			return true;
		}
		break;
	}

	if (!next_node(BAD_CAST STR_ITEM)) return false;

	return true;
}

bool
CSerializableXmlFile::write_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	pop_node();

	return true;
}

bool
CSerializableXmlFile::read_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	return true;
}

bool
CSerializableXmlFile::write_sgserializable_begin_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	if (*sgserializable_name == '\0') {
		if (xmlNewProp(m_stack_stream.back(), BAD_CAST STR_PROP_IS_NULL,
					   BAD_CAST STR_TRUE) == NULL) return false;
		return true;
	}

	if (xmlNewProp(m_stack_stream.back(),
				   BAD_CAST STR_PROP_INSTANCE_NAME,
				   BAD_CAST sgserializable_name) == NULL) return false;

	if (generic != PT_NOT_GENERIC) {
		string_t buf;
		TSGDataType::ptype_to_string(buf, generic, STRING_LEN);
		if (xmlNewProp(m_stack_stream.back(),
					   BAD_CAST STR_PROP_GENERIC_NAME, BAD_CAST buf)
			== NULL) return false;
	}

	return true;
}

bool
CSerializableXmlFile::read_sgserializable_begin_wrapped(
	const TSGDataType* type, char* sgserializable_name,
	EPrimitiveType* generic)
{
	xmlChar* buf;

	if ((buf = xmlGetProp(m_stack_stream.back(), BAD_CAST
						  STR_PROP_IS_NULL)) != NULL) {
		xmlFree(buf);
		*sgserializable_name = '\0';
		return true;
	}

	if ((buf = xmlGetProp(m_stack_stream.back(), BAD_CAST
						  STR_PROP_INSTANCE_NAME)) == NULL)
		return false;
	strncpy(sgserializable_name, (const char*) buf, STRING_LEN);
	xmlFree(buf);

	if ((buf = xmlGetProp(m_stack_stream.back(), BAD_CAST
						  STR_PROP_GENERIC_NAME)) != NULL) {
		if (!TSGDataType::string_to_ptype(generic, (const char*) buf))
			return false;
		xmlFree(buf);
	}

	return true;
}

bool
CSerializableXmlFile::write_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	return true;
}

bool
CSerializableXmlFile::read_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	return true;
}

bool
CSerializableXmlFile::write_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	if (!push_node(BAD_CAST name)) return false;

	string_t buf;
	type->to_string(buf, STRING_LEN);
	if (xmlNewProp(m_stack_stream.back(), BAD_CAST STR_PROP_TYPE,
				   BAD_CAST buf) == NULL) return false;

	return true;
}

bool
CSerializableXmlFile::read_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	bool result = true;

	if (!join_node(BAD_CAST name)) return false;

	string_t buf; type->to_string(buf, STRING_LEN);
	xmlChar* t;
	if ((t = xmlGetProp(m_stack_stream.back(), BAD_CAST STR_PROP_TYPE)
			) == NULL) return false;
	if (xmlStrcmp(BAD_CAST buf, t) != 0) result = false;
	xmlFree(t); if (!result) return false;

	return true;
}

bool
CSerializableXmlFile::write_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	pop_node();

	return true;
}

bool
CSerializableXmlFile::read_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	pop_node();

	return true;
}

#endif /* HAVE_XML  */
