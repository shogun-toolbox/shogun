/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <lib/config.h>
#ifdef HAVE_XML

#include <lib/common.h>
#include <io/SerializableXmlReader00.h>

using namespace shogun;

SerializableXmlReader00::SerializableXmlReader00(
	CSerializableXmlFile* file) { m_file = file; }

SerializableXmlReader00::~SerializableXmlReader00() {}

bool
SerializableXmlReader00::read_scalar_wrapped(
	const TSGDataType* type, void* param)
{
	xmlNode* m = m_file->m_stack_stream.back();

	bool result = true;
	xmlChar* xml_buf;
	if ((xml_buf = xmlNodeGetContent(m)) == NULL) return false;
	const char* buf = (const char*) xml_buf;

	switch (type->m_ptype) {
	case PT_BOOL:
		string_t bool_buf;

		if (sscanf(buf, "%" STRING_LEN_STR "s", bool_buf) != 1)
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
		if (sscanf(buf, "%" SCNi8, (int8_t*) param) != 1)
			result = false;
		break;
	case PT_UINT8:
		if (sscanf(buf, "%" SCNu8, (uint8_t*) param) != 1)
			result = false;
		break;
	case PT_INT16:
		if (sscanf(buf, "%" SCNi16, (int16_t*) param) != 1)
			result = false;
		break;
	case PT_UINT16:
		if (sscanf(buf, "%" SCNu16, (uint16_t*) param) != 1)
			result = false;
		break;
	case PT_INT32:
		if (sscanf(buf, "%" SCNi32, (int32_t*) param) != 1)
			result = false;
		break;
	case PT_UINT32:
		if (sscanf(buf, "%" SCNu32, (uint32_t*) param) != 1)
			result = false;
		break;
	case PT_INT64:
		if (sscanf(buf, "%" SCNi64, (int64_t*) param) != 1)
			result = false;
		break;
	case PT_UINT64:
		if (sscanf(buf, "%" SCNu64, (uint64_t*) param) != 1)
			result = false;
		break;
	case PT_FLOAT32:
		if (sscanf(buf, "%g", (float32_t*) param) != 1)
			result = false;
		break;
	case PT_FLOAT64:
		if (sscanf(buf, "%lg", (float64_t*) param) != 1)
			result = false;
		break;
	case PT_FLOATMAX:
		if (sscanf(buf, "%Lg", (floatmax_t*) param) != 1)
			result = false;
		break;
	case PT_COMPLEX128:
		float64_t c_real, c_imag;
		if (sscanf(buf, "(%lg,%lg)", &c_real, &c_imag) != 2)
			result = false;
#if defined(HAVE_CXX0X) || defined(HAVE_CXX11) || defined(_LIBCPP_VERSION)
		((complex128_t*) param)->real(c_real);
		((complex128_t*) param)->imag(c_imag);
#else
		((complex128_t*) param)->real()=c_real;
		((complex128_t*) param)->imag()=c_imag;
#endif
		break;
	case PT_UNDEFINED:
	case PT_SGOBJECT:
		SG_ERROR("read_scalar_wrapped(): Implementation error during"
				 " reading XmlFile!");
		result = false;
	}

	xmlFree(xml_buf);
	return result;
}

bool
SerializableXmlReader00::read_cont_begin_wrapped(
	const TSGDataType* type, index_t* len_read_y, index_t* len_read_x)
{
	xmlNode* m = m_file->m_stack_stream.back();

	switch (type->m_ctype) {
	case CT_NDARRAY:
		SG_NOTIMPLEMENTED
	case CT_SCALAR: break;
	case CT_VECTOR: case CT_SGVECTOR:
		*len_read_y = xmlChildElementCount(m);
		break;
	case CT_MATRIX: case CT_SGMATRIX:
		*len_read_x = xmlChildElementCount(m);

		for (xmlNode* cur=m->children; cur != NULL; cur=cur->next) {
			if (cur->type != XML_ELEMENT_NODE) continue;

			if (*len_read_y == 0)
				*len_read_y = xmlChildElementCount(cur);

			if (*len_read_y != (index_t) xmlChildElementCount(cur))
				return false;
		}
		break;
	case CT_UNDEFINED:
		SG_ERROR("type undefined\n");
	}

	return true;
}

bool
SerializableXmlReader00::read_cont_end_wrapped(
	const TSGDataType* type, index_t len_read_y, index_t len_read_x)
{
	if (len_read_y > 0) m_file->pop_node();

	if (type->m_ctype==CT_MATRIX || type->m_ctype==CT_SGMATRIX)
	{
		if (len_read_y*len_read_x>0)
			m_file->pop_node();
	}

	return true;
}

bool
SerializableXmlReader00::read_string_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	xmlNode* m = m_file->m_stack_stream.back();

	*length = xmlChildElementCount(m);

	return true;
}

bool
SerializableXmlReader00::read_string_end_wrapped(
	const TSGDataType* type, index_t length)
{
	if (length > 0) m_file->pop_node();

	return true;
}

bool
SerializableXmlReader00::read_stringentry_begin_wrapped(
	const TSGDataType* type, index_t y)
{
	if (y == 0) {
		if (!m_file->join_node(BAD_CAST STR_STRING)) return false;
		return true;
	}

	if (!m_file->next_node(BAD_CAST STR_STRING)) return false;

	return true;
}

bool
SerializableXmlReader00::read_stringentry_end_wrapped(
	const TSGDataType* type, index_t y)
{
	return true;
}

bool
SerializableXmlReader00::read_sparse_begin_wrapped(
	const TSGDataType* type, index_t* length)
{
	return true;
}

bool
SerializableXmlReader00::read_sparse_end_wrapped(
	const TSGDataType* type, index_t length)
{
	if (length > 0) m_file->pop_node();

	return true;
}

bool
SerializableXmlReader00::read_sparseentry_begin_wrapped(
	const TSGDataType* type, SGSparseVectorEntry<char>* first_entry,
	index_t* feat_index, index_t y)
{
	bool result = true;
	xmlChar* buf;

	if (y == 0) {
		if (!m_file->join_node(BAD_CAST STR_SPARSE)) return false;
	} else {
		if (!m_file->next_node(BAD_CAST STR_SPARSE)) return false;
	}

	if ((buf = xmlGetProp(m_file->m_stack_stream.back(), BAD_CAST
						  STR_PROP_FEATINDEX)) == NULL) return false;
	if (sscanf((const char*) buf, "%" PRIi32, feat_index) != 1)
		result = false;
	xmlFree(buf); if (!result) return false;

	return true;
}

bool
SerializableXmlReader00::read_sparseentry_end_wrapped(
	const TSGDataType* type, SGSparseVectorEntry<char>* first_entry,
	index_t* feat_index, index_t y)
{
	return true;
}

bool
SerializableXmlReader00::read_item_begin_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	switch (type->m_ctype) {
	case CT_NDARRAY:
		SG_NOTIMPLEMENTED
	case CT_SCALAR: break;
	case CT_VECTOR: case CT_SGVECTOR:
		if (y == 0) {
			if (!m_file->join_node(BAD_CAST STR_ITEM)) return false;
			return true;
		}
		break;
	case CT_MATRIX: case CT_SGMATRIX:
		if (y==0)
		{
			if (x != 0) { m_file->pop_node(); m_file->pop_node(); }

			string_t buf_x; snprintf(buf_x, STRING_LEN, "x%" PRIi32, x);
			if (!m_file->join_node(BAD_CAST buf_x)) return false;
			if (!m_file->join_node(BAD_CAST STR_ITEM)) return false;
			return true;
		}
		break;
	case CT_UNDEFINED:
		SG_ERROR("type undefined\n");
	}

	if (!m_file->next_node(BAD_CAST STR_ITEM)) return false;

	return true;
}

bool
SerializableXmlReader00::read_item_end_wrapped(
	const TSGDataType* type, index_t y, index_t x)
{
	return true;
}

bool
SerializableXmlReader00::read_sgserializable_begin_wrapped(
	const TSGDataType* type, char* sgserializable_name,
	EPrimitiveType* generic)
{
	xmlNode* m = m_file->m_stack_stream.back();
	xmlChar* buf;

	if ((buf = xmlGetProp(m, BAD_CAST STR_PROP_IS_NULL)) != NULL) {
		xmlFree(buf);
		*sgserializable_name = '\0';
		return true;
	}

	if ((buf = xmlGetProp(m, BAD_CAST STR_PROP_INSTANCE_NAME)) == NULL)
		return false;
	strncpy(sgserializable_name, (const char*) buf, STRING_LEN);
	xmlFree(buf);

	if ((buf = xmlGetProp(m, BAD_CAST STR_PROP_GENERIC_NAME))
		!= NULL) {
		if (!TSGDataType::string_to_ptype(generic, (const char*) buf))
			return false;
		xmlFree(buf);
	}

	return true;
}

bool
SerializableXmlReader00::read_sgserializable_end_wrapped(
	const TSGDataType* type, const char* sgserializable_name,
	EPrimitiveType generic)
{
	return true;
}

bool
SerializableXmlReader00::read_type_begin_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	bool result = true;

	SG_SET_LOCALE_C;

	if (!m_file->join_node(BAD_CAST name)) return false;

	string_t buf; type->to_string(buf, STRING_LEN);
	xmlChar* t;
	if ((t = xmlGetProp(m_file->m_stack_stream.back(),
						BAD_CAST STR_PROP_TYPE)) == NULL) return false;
	if (xmlStrcmp(BAD_CAST buf, t) != 0) result = false;
	xmlFree(t); if (!result) return false;

	return true;
}

bool
SerializableXmlReader00::read_type_end_wrapped(
	const TSGDataType* type, const char* name, const char* prefix)
{
	m_file->pop_node();

	SG_RESET_LOCALE;

	return true;
}

#endif /* HAVE_XML  */
