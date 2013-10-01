/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/lib/config.h>
#ifdef HAVE_XML

#include <shogun/io/SerializableXmlFile.h>
#include <shogun/io/SerializableXmlReader00.h>

#define STR_ROOT_NAME_00 \
	"_SHOGUN_SERIALIZABLE_XML_FILE_V_00_"

using namespace shogun;

CSerializableXmlFile::CSerializableXmlFile()
	:CSerializableFile() { init(false); }

CSerializableXmlFile::CSerializableXmlFile(const char* fname, char rw,
										   bool format)
	:CSerializableFile()
{
	CSerializableFile::init(NULL, rw, fname);
	init(format);
}

CSerializableXmlFile::~CSerializableXmlFile()
{
	close();
}

CSerializableFile::TSerializableReader*
CSerializableXmlFile::new_reader(char* dest_version, size_t n)
{
	xmlChar* name;

	if ((name = xmlGetNodePath(m_stack_stream.back())) == NULL)
		return NULL;

	strncpy(dest_version, (const char*) (name+1), n);
	xmlFree(name);

	if (strcmp(STR_ROOT_NAME_00, dest_version) == 0)
		return new SerializableXmlReader00(this);

	return NULL;
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
CSerializableXmlFile::pop_node()
{
	m_stack_stream.pop_back();
}

void
CSerializableXmlFile::init(bool format)
{
	m_format = format, m_doc = NULL;

	LIBXML_TEST_VERSION;

	if (m_filename == NULL || *m_filename == '\0') {
		SG_WARNING("Filename not given for opening file!\n")
		close(); return;
	}

	SG_DEBUG("Opening '%s'\n", m_filename)

	xmlNode* tmp;
	switch (m_task) {
	case 'r':
		if ((m_doc = xmlReadFile(m_filename, NULL, XML_PARSE_HUGE | XML_PARSE_NONET)) == NULL
			|| (tmp = xmlDocGetRootElement(m_doc)) == NULL)
		{
			SG_WARNING("Could not open file `%s' for reading!\n", m_filename)
			close(); return;
		}
		m_stack_stream.push_back(tmp);
		break;
	case 'w':
		m_doc = xmlNewDoc(BAD_CAST XML_DEFAULT_VERSION);
		m_stack_stream.push_back(xmlNewNode(
									 NULL, BAD_CAST STR_ROOT_NAME_00));
		xmlDocSetRootElement(m_doc, m_stack_stream.back());
		break;
	default:
		SG_WARNING("Could not open file `%s', unknown mode!\n",
				   m_filename);
		close(); return;
	}
}

void
CSerializableXmlFile::close()
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
CSerializableXmlFile::is_opened()
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
		if (snprintf(buf, STRING_LEN, "%" PRIi8, *(int8_t*) param
				) <= 0) return false;
		break;
	case PT_UINT8:
		if (snprintf(buf, STRING_LEN, "%" PRIu8, *(uint8_t*) param
				) <= 0) return false;
		break;
	case PT_INT16:
		if (snprintf(buf, STRING_LEN, "%" PRIi16, *(int16_t*) param
				) <= 0) return false;
		break;
	case PT_UINT16:
		if (snprintf(buf, STRING_LEN, "%" PRIu16, *(uint16_t*) param
				) <= 0) return false;
		break;
	case PT_INT32:
		if (snprintf(buf, STRING_LEN, "%" PRIi32, *(int32_t*) param
				) <= 0) return false;
		break;
	case PT_UINT32:
		if (snprintf(buf, STRING_LEN, "%" PRIu32, *(uint32_t*) param
				) <= 0) return false;
		break;
	case PT_INT64:
		if (snprintf(buf, STRING_LEN, "%" PRIi64, *(int64_t*) param
				) <= 0) return false;
		break;
	case PT_UINT64:
		if (snprintf(buf, STRING_LEN, "%" PRIu64, *(uint64_t*) param
				) <= 0) return false;
		break;
	case PT_FLOAT32:
		if (snprintf(buf, STRING_LEN, "%.16g", *(float32_t*) param
				) <= 0) return false;
		break;
	case PT_FLOAT64:
		if (snprintf(buf, STRING_LEN, "%.16lg", *(float64_t*) param
				) <= 0) return false;
		break;
	case PT_FLOATMAX:
		if (snprintf(buf, STRING_LEN, "%.16Lg", *(floatmax_t*)
					 param) <= 0) return false;
		break;
	case PT_COMPLEX128:
		if (snprintf(buf, STRING_LEN, "(%.16lg,%.16lg)", 
				((complex128_t*) param)->real(),((complex128_t*) param)->imag()
				) <= 0) return false;
		break;
	case PT_UNDEFINED:
	case PT_SGOBJECT:
		SG_ERROR("write_scalar_wrapped(): Implementation error during"
				 " writing XmlFile!");
		return false;
	}

	xmlNodeAddContent(m_stack_stream.back(), BAD_CAST buf);
	return true;
}

bool
CSerializableXmlFile::write_cont_begin_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	return true;
}

bool
CSerializableXmlFile::write_cont_end_wrapped(
	const TSGDataType* type, index_t len_real_y, index_t len_real_x)
{
	if (type->m_ctype==CT_MATRIX || type->m_ctype==CT_SGMATRIX)
		if (len_real_y*len_real_x>0)
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
CSerializableXmlFile::write_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
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
CSerializableXmlFile::write_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	pop_node();

	return true;
}

bool
CSerializableXmlFile::write_sparse_begin_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
CSerializableXmlFile::write_sparse_end_wrapped(
	const TSGDataType* type, index_t length)
{
	return true;
}

bool
CSerializableXmlFile::write_sparseentry_begin_wrapped(
	const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
	index_t feat_index, index_t y)
{
	push_node(BAD_CAST STR_SPARSE);

	string_t buf;
	snprintf(buf, STRING_LEN, "%" PRIi32, feat_index);
	if (xmlNewProp(m_stack_stream.back(), BAD_CAST STR_PROP_FEATINDEX,
				   BAD_CAST buf) == NULL) return false;
	return true;
}

bool
CSerializableXmlFile::write_sparseentry_end_wrapped(
	const TSGDataType* type, const SGSparseVectorEntry<char>* first_entry,
	index_t feat_index, index_t y)
{
	pop_node();

	return true;
}

bool
CSerializableXmlFile::write_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	if (type->m_ctype==CT_MATRIX || type->m_ctype==CT_SGMATRIX) {
		if (y==0)
		{
			if (x != 0) pop_node();

			string_t buf_x; snprintf(buf_x, STRING_LEN, "x%" PRIi32, x);
			if (!push_node(BAD_CAST buf_x)) return false;
		}
	}

	push_node(BAD_CAST STR_ITEM);

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
CSerializableXmlFile::write_sgserializable_end_wrapped(
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

	SG_SET_LOCALE_C;

	string_t buf;
	type->to_string(buf, STRING_LEN);
	if (xmlNewProp(m_stack_stream.back(), BAD_CAST STR_PROP_TYPE,
				   BAD_CAST buf) == NULL) return false;

	return true;
}

bool
CSerializableXmlFile::write_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	pop_node();

	SG_RESET_LOCALE;

	return true;
}

#endif /* HAVE_XML  */
