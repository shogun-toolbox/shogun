/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Soumyajit De, Jacob Walker,
 *          Thoralf Klein, Sergey Lisitsyn, Bjoern Esser, Viktor Gal,
 *          Weijie Lin, Yori Zwols, Leon Kuchenbecker
 */

#include <cctype>
#include <string.h>

#include <shogun/base/Parameter.h>
#include <shogun/base/class_list.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/Hash.h>
#include <shogun/lib/common.h>
#include <shogun/lib/memory.h>
#include <shogun/mathematics/Math.h>

#include <shogun/io/SerializableFile.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

TParameter::TParameter(
    const TSGDataType* datatype, void* parameter, const char* name,
    const char* description)
    : m_datatype(*datatype)
{
	m_parameter = parameter;
	m_name = get_strdup(name);
	m_description = get_strdup(description);
}

TParameter::~TParameter()
{
	SG_FREE(m_description);
	SG_FREE(m_name);
}

char* TParameter::new_prefix(const char* s1, const char* s2)
{
	char* tmp = SG_MALLOC(char, strlen(s1) + strlen(s2) + 2);

	sprintf(tmp, "%s%s/", s1, s2);

	return tmp;
}

void TParameter::print(const char* prefix)
{
	string_t buf;
	m_datatype.to_string(buf, STRING_LEN);

	SG_SPRINT(
	    "\n%s\n%35s %24s :%s\n", prefix,
	    m_description == NULL || *m_description == '\0' ? "(Parameter)"
	                                                    : m_description,
	    m_name, buf);

	if (m_datatype.m_ptype == PT_SGOBJECT && m_datatype.m_stype == ST_NONE &&
	    m_datatype.m_ctype == CT_SCALAR && *(CSGObject**)m_parameter != NULL)
	{
		char* p = new_prefix(prefix, m_name);
		(*(CSGObject**)m_parameter)->print_serializable(p);
		SG_FREE(p);
	}
}

void TParameter::delete_cont()
{
	if (*(void**)m_parameter != NULL)
	{
		index_t old_length = m_datatype.m_length_y ? *m_datatype.m_length_y : 0;
		switch (m_datatype.m_ctype)
		{
		case CT_NDARRAY:
			SG_SNOTIMPLEMENTED
			break;
		case CT_MATRIX:
		case CT_SGMATRIX:
			old_length *= *m_datatype.m_length_x;
			break;
		case CT_SCALAR:
		case CT_VECTOR:
		case CT_SGVECTOR:
			break;
		case CT_UNDEFINED:
		default:
			SG_SERROR("Implementation error: undefined container type\n");
			break;
		}

		switch (m_datatype.m_stype)
		{
		case ST_NONE:
			switch (m_datatype.m_ptype)
			{
			case PT_BOOL:
				SG_FREE(*(bool**)m_parameter);
				break;
			case PT_CHAR:
				SG_FREE(*(char**)m_parameter);
				break;
			case PT_INT8:
				SG_FREE(*(int8_t**)m_parameter);
				break;
			case PT_UINT8:
				SG_FREE(*(uint8_t**)m_parameter);
				break;
			case PT_INT16:
				SG_FREE(*(int16_t**)m_parameter);
				break;
			case PT_UINT16:
				SG_FREE(*(uint16_t**)m_parameter);
				break;
			case PT_INT32:
				SG_FREE(*(int32_t**)m_parameter);
				break;
			case PT_UINT32:
				SG_FREE(*(uint32_t**)m_parameter);
				break;
			case PT_INT64:
				SG_FREE(*(int64_t**)m_parameter);
				break;
			case PT_UINT64:
				SG_FREE(*(uint64_t**)m_parameter);
				break;
			case PT_FLOAT32:
				SG_FREE(*(float32_t**)m_parameter);
				break;
			case PT_FLOAT64:
				SG_FREE(*(float64_t**)m_parameter);
				break;
			case PT_FLOATMAX:
				SG_FREE(*(floatmax_t**)m_parameter);
				break;
			case PT_COMPLEX128:
				SG_FREE(*(complex128_t**)m_parameter);
				break;
			case PT_SGOBJECT:
			{
				CSGObject** buf = *(CSGObject***)m_parameter;

				for (index_t i = 0; i < old_length; i++)
					SG_UNREF(buf[i]);

				SG_FREE(buf);
				break;
			}
			case PT_UNDEFINED:
			default:
				SG_SERROR("Implementation error: undefined primitive type\n");
				break;
			}
			break;
		case ST_STRING:
		{
			for (index_t i = 0; i < old_length; i++)
			{
				SGString<char>* buf =
				    (SGString<
				        char>*)(*(char**)m_parameter + i * m_datatype.sizeof_stype());
				if (buf->slen > 0)
					SG_FREE(buf->string);
				break;
			}
		}

			switch (m_datatype.m_ptype)
			{
			case PT_BOOL:
				SG_FREE(*(SGString<bool>**)m_parameter);
				break;
			case PT_CHAR:
				SG_FREE(*(SGString<char>**)m_parameter);
				break;
			case PT_INT8:
				SG_FREE(*(SGString<int8_t>**)m_parameter);
				break;
			case PT_UINT8:
				SG_FREE(*(SGString<uint8_t>**)m_parameter);
				break;
			case PT_INT16:
				SG_FREE(*(SGString<int16_t>**)m_parameter);
				break;
			case PT_UINT16:
				SG_FREE(*(SGString<uint16_t>**)m_parameter);
				break;
			case PT_INT32:
				SG_FREE(*(SGString<int32_t>**)m_parameter);
				break;
			case PT_UINT32:
				SG_FREE(*(SGString<uint32_t>**)m_parameter);
				break;
			case PT_INT64:
				SG_FREE(*(SGString<int64_t>**)m_parameter);
				break;
			case PT_UINT64:
				SG_FREE(*(SGString<uint64_t>**)m_parameter);
				break;
			case PT_FLOAT32:
				SG_FREE(*(SGString<float32_t>**)m_parameter);
				break;
			case PT_FLOAT64:
				SG_FREE(*(SGString<float64_t>**)m_parameter);
				break;
			case PT_FLOATMAX:
				SG_FREE(*(SGString<floatmax_t>**)m_parameter);
				break;
			case PT_COMPLEX128:
				SG_SERROR("TParameter::delete_cont(): Parameters of strings"
				          " of complex128_t are not supported");
				break;
			case PT_SGOBJECT:
				SG_SERROR("TParameter::delete_cont(): Implementation "
				          "error: Could not delete "
				          "String<SGSerializable*>");
				break;
			case PT_UNDEFINED:
			default:
				SG_SERROR("Implementation error: undefined primitive type\n");
				break;
			}
			break;
		case ST_SPARSE:
			for (index_t i = 0; i < old_length; i++)
			{
				SGSparseVector<char>* buf =
				    (SGSparseVector<
				        char>*)(*(char**)m_parameter + i * m_datatype.sizeof_stype());
				if (buf->num_feat_entries > 0)
					SG_FREE(buf->features);
			}

			switch (m_datatype.m_ptype)
			{
			case PT_BOOL:
				SG_FREE(*(SGSparseVector<bool>**)m_parameter);
				break;
			case PT_CHAR:
				SG_FREE(*(SGSparseVector<char>**)m_parameter);
				break;
			case PT_INT8:
				SG_FREE(*(SGSparseVector<int8_t>**)m_parameter);
				break;
			case PT_UINT8:
				SG_FREE(*(SGSparseVector<uint8_t>**)m_parameter);
				break;
			case PT_INT16:
				SG_FREE(*(SGSparseVector<int16_t>**)m_parameter);
				break;
			case PT_UINT16:
				SG_FREE(*(SGSparseVector<uint16_t>**)m_parameter);
				break;
			case PT_INT32:
				SG_FREE(*(SGSparseVector<int32_t>**)m_parameter);
				break;
			case PT_UINT32:
				SG_FREE(*(SGSparseVector<uint32_t>**)m_parameter);
				break;
			case PT_INT64:
				SG_FREE(*(SGSparseVector<int64_t>**)m_parameter);
				break;
			case PT_UINT64:
				SG_FREE(*(SGSparseVector<uint64_t>**)m_parameter);
				break;
			case PT_FLOAT32:
				SG_FREE(*(SGSparseVector<float32_t>**)m_parameter);
				break;
			case PT_FLOAT64:
				SG_FREE(*(SGSparseVector<float64_t>**)m_parameter);
				break;
			case PT_FLOATMAX:
				SG_FREE(*(SGSparseVector<floatmax_t>**)m_parameter);
				break;
			case PT_COMPLEX128:
				SG_FREE(*(SGSparseVector<complex128_t>**)m_parameter);
				break;
			case PT_SGOBJECT:
				SG_SERROR("TParameter::delete_cont(): Implementation "
				          "error: Could not delete "
				          "Sparse<SGSerializable*>");
				break;
			case PT_UNDEFINED:
			default:
				SG_SERROR("Implementation error: undefined primitive type\n");
				break;
			}
			break;
		case ST_UNDEFINED:
		default:
			SG_SERROR("Implementation error: undefined structure type\n");
			break;
		} /* switch (m_datatype.m_stype)  */
	}     /* if (*(void**) m_parameter != NULL)  */

	*(void**)m_parameter = NULL;
}

void TParameter::new_cont(SGVector<index_t> dims)
{
	char* s = SG_MALLOC(char, 200);
	m_datatype.to_string(s, 200);
	SG_SDEBUG(
	    "entering TParameter::new_cont for \"%s\" of type %s\n", s,
	    m_name ? m_name : "(nil)");
	SG_FREE(s);
	delete_cont();

	index_t new_length = dims.product();
	if (new_length == 0)
		return;

	switch (m_datatype.m_stype)
	{
	case ST_NONE:
		switch (m_datatype.m_ptype)
		{
		case PT_BOOL:
			*(bool**)m_parameter = SG_MALLOC(bool, new_length);
			break;
		case PT_CHAR:
			*(char**)m_parameter = SG_MALLOC(char, new_length);
			break;
		case PT_INT8:
			*(int8_t**)m_parameter = SG_MALLOC(int8_t, new_length);
			break;
		case PT_UINT8:
			*(uint8_t**)m_parameter = SG_MALLOC(uint8_t, new_length);
			break;
		case PT_INT16:
			*(int16_t**)m_parameter = SG_MALLOC(int16_t, new_length);
			break;
		case PT_UINT16:
			*(uint16_t**)m_parameter = SG_MALLOC(uint16_t, new_length);
			break;
		case PT_INT32:
			*(int32_t**)m_parameter = SG_MALLOC(int32_t, new_length);
			break;
		case PT_UINT32:
			*(uint32_t**)m_parameter = SG_MALLOC(uint32_t, new_length);
			break;
		case PT_INT64:
			*(int64_t**)m_parameter = SG_MALLOC(int64_t, new_length);
			break;
		case PT_UINT64:
			*(uint64_t**)m_parameter = SG_MALLOC(uint64_t, new_length);
			break;
		case PT_FLOAT32:
			*(float32_t**)m_parameter = SG_MALLOC(float32_t, new_length);
			break;
		case PT_FLOAT64:
			*(float64_t**)m_parameter = SG_MALLOC(float64_t, new_length);
			break;
		case PT_FLOATMAX:
			*(floatmax_t**)m_parameter = SG_MALLOC(floatmax_t, new_length);
			break;
		case PT_COMPLEX128:
			*(complex128_t**)m_parameter = SG_MALLOC(complex128_t, new_length);
			break;
		case PT_SGOBJECT:
			*(CSGObject***)m_parameter = SG_CALLOC(CSGObject*, new_length);
			break;
		case PT_UNDEFINED:
		default:
			SG_SERROR("Implementation error: undefined primitive type\n");
			break;
		}
		break;
	case ST_STRING:
		switch (m_datatype.m_ptype)
		{
		case PT_BOOL:
			*(SGString<bool>**)m_parameter =
			    SG_MALLOC(SGString<bool>, new_length);
			break;
		case PT_CHAR:
			*(SGString<char>**)m_parameter =
			    SG_MALLOC(SGString<char>, new_length);
			break;
		case PT_INT8:
			*(SGString<int8_t>**)m_parameter =
			    SG_MALLOC(SGString<int8_t>, new_length);
			break;
		case PT_UINT8:
			*(SGString<uint8_t>**)m_parameter =
			    SG_MALLOC(SGString<uint8_t>, new_length);
			break;
		case PT_INT16:
			*(SGString<int16_t>**)m_parameter =
			    SG_MALLOC(SGString<int16_t>, new_length);
			break;
		case PT_UINT16:
			*(SGString<uint16_t>**)m_parameter =
			    SG_MALLOC(SGString<uint16_t>, new_length);
			break;
		case PT_INT32:
			*(SGString<int32_t>**)m_parameter =
			    SG_MALLOC(SGString<int32_t>, new_length);
			break;
		case PT_UINT32:
			*(SGString<uint32_t>**)m_parameter =
			    SG_MALLOC(SGString<uint32_t>, new_length);
			break;
		case PT_INT64:
			*(SGString<int64_t>**)m_parameter =
			    SG_MALLOC(SGString<int64_t>, new_length);
			break;
		case PT_UINT64:
			*(SGString<uint64_t>**)m_parameter =
			    SG_MALLOC(SGString<uint64_t>, new_length);
			break;
		case PT_FLOAT32:
			*(SGString<float32_t>**)m_parameter =
			    SG_MALLOC(SGString<float32_t>, new_length);
			break;
		case PT_FLOAT64:
			*(SGString<float64_t>**)m_parameter =
			    SG_MALLOC(SGString<float64_t>, new_length);
			break;
		case PT_FLOATMAX:
			*(SGString<floatmax_t>**)m_parameter =
			    SG_MALLOC(SGString<floatmax_t>, new_length);
			break;
		case PT_COMPLEX128:
			SG_SERROR("TParameter::new_cont(): Implementation "
			          "error: Could not allocate "
			          "String<complex128>");
			break;
		case PT_SGOBJECT:
			SG_SERROR("TParameter::new_cont(): Implementation "
			          "error: Could not allocate "
			          "String<SGSerializable*>");
			break;
		case PT_UNDEFINED:
		default:
			SG_SERROR("Implementation error: undefined primitive type\n");
			break;
		}
		memset(*(void**)m_parameter, 0, new_length * m_datatype.sizeof_stype());
		break;
	case ST_SPARSE:
		switch (m_datatype.m_ptype)
		{
		case PT_BOOL:
			*(SGSparseVector<bool>**)m_parameter =
			    SG_MALLOC(SGSparseVector<bool>, new_length);
			break;
		case PT_CHAR:
			*(SGSparseVector<char>**)m_parameter =
			    SG_MALLOC(SGSparseVector<char>, new_length);
			break;
		case PT_INT8:
			*(SGSparseVector<int8_t>**)m_parameter =
			    SG_MALLOC(SGSparseVector<int8_t>, new_length);
			break;
		case PT_UINT8:
			*(SGSparseVector<uint8_t>**)m_parameter =
			    SG_MALLOC(SGSparseVector<uint8_t>, new_length);
			break;
		case PT_INT16:
			*(SGSparseVector<int16_t>**)m_parameter =
			    SG_MALLOC(SGSparseVector<int16_t>, new_length);
			break;
		case PT_UINT16:
			*(SGSparseVector<uint16_t>**)m_parameter =
			    SG_MALLOC(SGSparseVector<uint16_t>, new_length);
			break;
		case PT_INT32:
			*(SGSparseVector<int32_t>**)m_parameter =
			    SG_MALLOC(SGSparseVector<int32_t>, new_length);
			break;
		case PT_UINT32:
			*(SGSparseVector<uint32_t>**)m_parameter =
			    SG_MALLOC(SGSparseVector<uint32_t>, new_length);
			break;
		case PT_INT64:
			*(SGSparseVector<int64_t>**)m_parameter =
			    SG_MALLOC(SGSparseVector<int64_t>, new_length);
			break;
		case PT_UINT64:
			*(SGSparseVector<uint64_t>**)m_parameter =
			    SG_MALLOC(SGSparseVector<uint64_t>, new_length);
			break;
		case PT_FLOAT32:
			*(SGSparseVector<float32_t>**)m_parameter =
			    SG_MALLOC(SGSparseVector<float32_t>, new_length);
			break;
		case PT_FLOAT64:
			*(SGSparseVector<float64_t>**)m_parameter =
			    SG_MALLOC(SGSparseVector<float64_t>, new_length);
			break;
		case PT_FLOATMAX:
			*(SGSparseVector<floatmax_t>**)m_parameter =
			    SG_MALLOC(SGSparseVector<floatmax_t>, new_length);
			break;
		case PT_COMPLEX128:
			*(SGSparseVector<complex128_t>**)m_parameter =
			    SG_MALLOC(SGSparseVector<complex128_t>, new_length);
			break;
		case PT_SGOBJECT:
			SG_SERROR("TParameter::new_cont(): Implementation "
			          "error: Could not allocate "
			          "Sparse<SGSerializable*>");
			break;
		case PT_UNDEFINED:
		default:
			SG_SERROR("Implementation error: undefined primitive type\n");
			break;
		}
		break;
	case ST_UNDEFINED:
	default:
		SG_SERROR("Implementation error: undefined structure type\n");
		break;
	} /* switch (m_datatype.m_stype)  */

	s = SG_MALLOC(char, 200);
	m_datatype.to_string(s, 200);
	SG_SDEBUG(
	    "leaving TParameter::new_cont for \"%s\" of type %s\n", s,
	    m_name ? m_name : "(nil)");
	SG_FREE(s);
}

bool TParameter::new_sgserial(
    CSGObject** param, EPrimitiveType generic, const char* sgserializable_name,
    const char* prefix)
{
	if (*param != NULL)
		SG_UNREF(*param);

	*param = create(sgserializable_name, generic);

	if (*param == NULL)
	{
		string_t buf = {'\0'};

		if (generic != PT_NOT_GENERIC)
		{
			buf[0] = '<';
			TSGDataType::ptype_to_string(buf + 1, generic, STRING_LEN - 3);
			strcat(buf, ">");
		}

		SG_SWARNING(
		    "TParameter::new_sgserial(): "
		    "Class `C%s%s' was not listed during compiling Shogun"
		    " :( ...  Can not construct it for `%s%s'!",
		    sgserializable_name, buf, prefix, m_name);

		return false;
	}

	SG_REF(*param);
	return true;
}

bool TParameter::save_ptype(
    CSerializableFile* file, const void* param, const char* prefix)
{
	if (m_datatype.m_ptype == PT_SGOBJECT)
	{
		const char* sgserial_name = "";
		EPrimitiveType generic = PT_NOT_GENERIC;

		if (*(CSGObject**)param != NULL)
		{
			sgserial_name = (*(CSGObject**)param)->get_name();
			(*(CSGObject**)param)->is_generic(&generic);
		}

		if (!file->write_sgserializable_begin(
		        &m_datatype, m_name, prefix, sgserial_name, generic))
			return false;
		if (*sgserial_name != '\0')
		{
			char* p = new_prefix(prefix, m_name);
			bool result = (*(CSGObject**)param)->save_serializable(file, p);
			SG_FREE(p);
			if (!result)
				return false;
		}
		if (!file->write_sgserializable_end(
		        &m_datatype, m_name, prefix, sgserial_name, generic))
			return false;
	}
	else if (!file->write_scalar(&m_datatype, m_name, prefix, param))
		return false;

	return true;
}

bool TParameter::load_ptype(
    CSerializableFile* file, void* param, const char* prefix)
{
	if (m_datatype.m_ptype == PT_SGOBJECT)
	{
		string_t sgserial_name = {'\0'};
		EPrimitiveType generic = PT_NOT_GENERIC;

		if (!file->read_sgserializable_begin(
		        &m_datatype, m_name, prefix, sgserial_name, &generic))
			return false;
		if (*sgserial_name != '\0')
		{
			if (!new_sgserial(
			        (CSGObject**)param, generic, sgserial_name, prefix))
				return false;

			char* p = new_prefix(prefix, m_name);
			bool result = (*(CSGObject**)param)->load_serializable(file, p);
			SG_FREE(p);
			if (!result)
				return false;
		}
		if (!file->read_sgserializable_end(
		        &m_datatype, m_name, prefix, sgserial_name, generic))
			return false;
	}
	else if (!file->read_scalar(&m_datatype, m_name, prefix, param))
		return false;

	return true;
}

bool TParameter::save_stype(
    CSerializableFile* file, const void* param, const char* prefix)
{
	SGString<char>* str_ptr = (SGString<char>*)param;
	SGSparseVector<char>* spr_ptr = (SGSparseVector<char>*)param;
	index_t len_real;

	switch (m_datatype.m_stype)
	{
	case ST_NONE:
		if (!save_ptype(file, param, prefix))
			return false;
		break;
	case ST_STRING:
		len_real = str_ptr->slen;
		if (str_ptr->string == NULL && len_real != 0)
		{
			SG_SWARNING(
			    "Inconsistency between data structure and "
			    "len during saving string `%s%s'!  Continuing"
			    " with len=0.\n",
			    prefix, m_name);
			len_real = 0;
		}
		if (!file->write_string_begin(&m_datatype, m_name, prefix, len_real))
			return false;
		for (index_t i = 0; i < len_real; i++)
		{
			if (!file->write_stringentry_begin(&m_datatype, m_name, prefix, i))
				return false;
			if (!save_ptype(
			        file,
			        (char*)str_ptr->string + i * m_datatype.sizeof_ptype(),
			        prefix))
				return false;
			if (!file->write_stringentry_end(&m_datatype, m_name, prefix, i))
				return false;
		}
		if (!file->write_string_end(&m_datatype, m_name, prefix, len_real))
			return false;
		break;
	case ST_SPARSE:
		len_real = spr_ptr->num_feat_entries;
		if (spr_ptr->features == NULL && len_real != 0)
		{
			SG_SWARNING(
			    "Inconsistency between data structure and "
			    "len during saving sparse `%s%s'!  Continuing"
			    " with len=0.\n",
			    prefix, m_name);
			len_real = 0;
		}
		if (!file->write_sparse_begin(&m_datatype, m_name, prefix, len_real))
			return false;
		for (index_t i = 0; i < len_real; i++)
		{
			SGSparseVectorEntry<char>* cur = (SGSparseVectorEntry<char>*)
				((char*) spr_ptr->features + i *TSGDataType
				 ::sizeof_sparseentry(m_datatype.m_ptype));
			if (!file->write_sparseentry_begin(
			        &m_datatype, m_name, prefix, spr_ptr->features,
			        cur->feat_index, i))
				return false;
			if (!save_ptype(
			        file,
			        (char*)cur +
			            TSGDataType ::offset_sparseentry(m_datatype.m_ptype),
			        prefix))
				return false;
			if (!file->write_sparseentry_end(
			        &m_datatype, m_name, prefix, spr_ptr->features,
			        cur->feat_index, i))
				return false;
		}
		if (!file->write_sparse_end(&m_datatype, m_name, prefix, len_real))
			return false;
		break;
	case ST_UNDEFINED:
	default:
		SG_SERROR("Implementation error: undefined structure type\n");
		break;
	}

	return true;
}

bool TParameter::load_stype(
    CSerializableFile* file, void* param, const char* prefix)
{
	SGString<char>* str_ptr = (SGString<char>*)param;
	SGSparseVector<char>* spr_ptr = (SGSparseVector<char>*)param;
	index_t len_real = 0;

	switch (m_datatype.m_stype)
	{
	case ST_NONE:
		if (!load_ptype(file, param, prefix))
			return false;
		break;
	case ST_STRING:
		if (!file->read_string_begin(&m_datatype, m_name, prefix, &len_real))
			return false;
		str_ptr->string =
		    len_real > 0 ? SG_MALLOC(char, len_real* m_datatype.sizeof_ptype())
		                 : NULL;
		for (index_t i = 0; i < len_real; i++)
		{
			if (!file->read_stringentry_begin(&m_datatype, m_name, prefix, i))
				return false;
			if (!load_ptype(
			        file,
			        (char*)str_ptr->string + i * m_datatype.sizeof_ptype(),
			        prefix))
				return false;
			if (!file->read_stringentry_end(&m_datatype, m_name, prefix, i))
				return false;
		}
		if (!file->read_string_end(&m_datatype, m_name, prefix, len_real))
			return false;
		str_ptr->slen = len_real;
		break;
	case ST_SPARSE:
		if (!file->read_sparse_begin(&m_datatype, m_name, prefix, &len_real))
			return false;
		spr_ptr->features =
		    len_real > 0 ? (SGSparseVectorEntry<char>*)SG_MALLOC(
		                       char, len_real* TSGDataType::sizeof_sparseentry(
		                                 m_datatype.m_ptype))
		                 : NULL;
		for (index_t i = 0; i < len_real; i++)
		{
			SGSparseVectorEntry<char>* cur = (SGSparseVectorEntry<char>*)
				((char*) spr_ptr->features + i *TSGDataType
				 ::sizeof_sparseentry(m_datatype.m_ptype));
			if (!file->read_sparseentry_begin(
			        &m_datatype, m_name, prefix, spr_ptr->features,
			        &cur->feat_index, i))
				return false;
			if (!load_ptype(
			        file,
			        (char*)cur +
			            TSGDataType ::offset_sparseentry(m_datatype.m_ptype),
			        prefix))
				return false;
			if (!file->read_sparseentry_end(
			        &m_datatype, m_name, prefix, spr_ptr->features,
			        &cur->feat_index, i))
				return false;
		}

		if (!file->read_sparse_end(&m_datatype, m_name, prefix, len_real))
			return false;

		spr_ptr->num_feat_entries = len_real;
		break;
	case ST_UNDEFINED:
	default:
		SG_SERROR("Implementation error: undefined structure type\n");
		break;
	}

	return true;
}

void TParameter::get_incremental_hash(
    uint32_t& hash, uint32_t& carry, uint32_t& total_length)
{

	switch (m_datatype.m_ctype)
	{
	case CT_NDARRAY:
		SG_SNOTIMPLEMENTED
		break;
	case CT_SCALAR:
	{
		uint8_t* data = ((uint8_t*)m_parameter);
		uint32_t size = m_datatype.sizeof_stype();
		total_length += size;
		CHash::IncrementalMurmurHash3(&hash, &carry, data, size);
		break;
	}
	case CT_VECTOR:
	case CT_MATRIX:
	case CT_SGVECTOR:
	case CT_SGMATRIX:
	{
		index_t len_real_y = 0, len_real_x = 0;

		if (m_datatype.m_length_y)
			len_real_y = *m_datatype.m_length_y;

		else
			len_real_y = 1;

		if (*(void**)m_parameter == NULL && len_real_y != 0)
		{
			SG_SWARNING(
			    "Inconsistency between data structure and "
			    "len_y during hashing `%s'!  Continuing with "
			    "len_y=0.\n",
			    m_name);
			len_real_y = 0;
		}

		switch (m_datatype.m_ctype)
		{
		case CT_NDARRAY:
			SG_SNOTIMPLEMENTED
			break;
		case CT_VECTOR:
		case CT_SGVECTOR:
			len_real_x = 1;
			break;
		case CT_MATRIX:
		case CT_SGMATRIX:
			len_real_x = *m_datatype.m_length_x;

			if (*(void**)m_parameter == NULL && len_real_x != 0)
			{
				SG_SWARNING(
				    "Inconsistency between data structure and "
				    "len_x during hashing %s'!  Continuing "
				    "with len_x=0.\n",
				    m_name);
				len_real_x = 0;
			}

			if (len_real_x * len_real_y == 0)
				len_real_x = len_real_y = 0;

			break;

		case CT_SCALAR:
			break;
		case CT_UNDEFINED:
		default:
			SG_SERROR("Implementation error: undefined container type\n");
			break;
		}
		uint32_t size = (len_real_x * len_real_y) * m_datatype.sizeof_stype();

		total_length += size;

		uint8_t* data = (*(uint8_t**)m_parameter);

		CHash::IncrementalMurmurHash3(&hash, &carry, data, size);
		break;
	}
	case CT_UNDEFINED:
	default:
		SG_SERROR("Implementation error: undefined container type\n");
		break;
	}
}

bool TParameter::is_valid()
{
	return m_datatype.get_num_elements() > 0;
}

bool TParameter::save(CSerializableFile* file, const char* prefix)
{
	const int32_t buflen = 100;
	char* buf = SG_MALLOC(char, buflen);
	m_datatype.to_string(buf, buflen);
	SG_SINFO("Saving parameter '%s' of type '%s'\n", m_name, buf)
	SG_FREE(buf);

	if (!file->write_type_begin(&m_datatype, m_name, prefix))
		return false;

	switch (m_datatype.m_ctype)
	{
	case CT_NDARRAY:
		SG_SNOTIMPLEMENTED
		break;
	case CT_SCALAR:
		if (!save_stype(file, m_parameter, prefix))
			return false;
		break;
	case CT_VECTOR:
	case CT_MATRIX:
	case CT_SGVECTOR:
	case CT_SGMATRIX:
	{
		index_t len_real_y = 0, len_real_x = 0;

		len_real_y = *m_datatype.m_length_y;
		if (*(void**)m_parameter == NULL && len_real_y != 0)
		{
			SG_SWARNING(
			    "Inconsistency between data structure and "
			    "len_y during saving `%s%s'!  Continuing with "
			    "len_y=0.\n",
			    prefix, m_name);
			len_real_y = 0;
		}

		switch (m_datatype.m_ctype)
		{
		case CT_NDARRAY:
			SG_SNOTIMPLEMENTED
			break;
		case CT_VECTOR:
		case CT_SGVECTOR:
			len_real_x = 1;
			break;
		case CT_MATRIX:
		case CT_SGMATRIX:
			len_real_x = *m_datatype.m_length_x;
			if (*(void**)m_parameter == NULL && len_real_x != 0)
			{
				SG_SWARNING(
				    "Inconsistency between data structure and "
				    "len_x during saving `%s%s'!  Continuing "
				    "with len_x=0.\n",
				    prefix, m_name);
				len_real_x = 0;
			}

			if (len_real_x * len_real_y == 0)
				len_real_x = len_real_y = 0;

			break;
		case CT_SCALAR:
			break;
		case CT_UNDEFINED:
		default:
			SG_SERROR("Implementation error: undefined container type\n");
			break;
		}

		if (!file->write_cont_begin(
		        &m_datatype, m_name, prefix, len_real_y, len_real_x))
			return false;

		/* ******************************************************** */

		for (index_t x = 0; x < len_real_x; x++)
			for (index_t y = 0; y < len_real_y; y++)
			{
				if (!file->write_item_begin(&m_datatype, m_name, prefix, y, x))
					return false;

				if (!save_stype(
				        file,
				        (*(char**)m_parameter) +
				            (x * len_real_y + y) * m_datatype.sizeof_stype(),
				        prefix))
					return false;
				if (!file->write_item_end(&m_datatype, m_name, prefix, y, x))
					return false;
			}

		/* ******************************************************** */

		if (!file->write_cont_end(
		        &m_datatype, m_name, prefix, len_real_y, len_real_x))
			return false;

		break;
	}
	case CT_UNDEFINED:
	default:
		SG_SERROR("Implementation error: undefined container type\n");
		break;
	}

	if (!file->write_type_end(&m_datatype, m_name, prefix))
		return false;

	return true;
}

bool TParameter::load(CSerializableFile* file, const char* prefix)
{
	REQUIRE(file != NULL, "Serializable file object should be != NULL\n");

	const int32_t buflen = 100;
	char* buf = SG_MALLOC(char, buflen);
	m_datatype.to_string(buf, buflen);
	SG_SDEBUG("Loading parameter '%s' of type '%s'\n", m_name, buf)
	SG_FREE(buf);

	if (!file->read_type_begin(&m_datatype, m_name, prefix))
		return false;

	switch (m_datatype.m_ctype)
	{
	case CT_NDARRAY:
		SG_SNOTIMPLEMENTED
		break;
	case CT_SCALAR:
		if (!load_stype(file, m_parameter, prefix))
			return false;
		break;

	case CT_VECTOR:
	case CT_MATRIX:
	case CT_SGVECTOR:
	case CT_SGMATRIX:
	{
		SGVector<index_t> dims(2);
		dims.zero();

		if (!file->read_cont_begin(
		        &m_datatype, m_name, prefix, &dims.vector[1], &dims.vector[0]))
			return false;

		switch (m_datatype.m_ctype)
		{
		case CT_NDARRAY:
			SG_SNOTIMPLEMENTED
			break;
		case CT_VECTOR:
		case CT_SGVECTOR:
			dims[0] = 1;
			new_cont(dims);
			break;
		case CT_MATRIX:
		case CT_SGMATRIX:
			new_cont(dims);
			break;
		case CT_SCALAR:
			break;
		case CT_UNDEFINED:
		default:
			SG_SERROR("Implementation error: undefined container type\n");
			break;
		}

		for (index_t x = 0; x < dims[0]; x++)
		{
			for (index_t y = 0; y < dims[1]; y++)
			{
				if (!file->read_item_begin(&m_datatype, m_name, prefix, y, x))
					return false;

				if (!load_stype(
				        file,
				        (*(char**)m_parameter) +
				            (x * dims[1] + y) * m_datatype.sizeof_stype(),
				        prefix))
					return false;
				if (!file->read_item_end(&m_datatype, m_name, prefix, y, x))
					return false;
			}
		}

		switch (m_datatype.m_ctype)
		{
		case CT_NDARRAY:
			SG_SNOTIMPLEMENTED
			break;
		case CT_VECTOR:
		case CT_SGVECTOR:
			*m_datatype.m_length_y = dims[1];
			break;
		case CT_MATRIX:
		case CT_SGMATRIX:
			*m_datatype.m_length_y = dims[1];
			*m_datatype.m_length_x = dims[0];
			break;
		case CT_SCALAR:
			break;
		case CT_UNDEFINED:
		default:
			SG_SERROR("Implementation error: undefined container type\n");
			break;
		}

		if (!file->read_cont_end(&m_datatype, m_name, prefix, dims[1], dims[0]))
			return false;

		break;
	}
	case CT_UNDEFINED:
	default:
		SG_SERROR("Implementation error: undefined container type\n");
		break;
	}

	if (!file->read_type_end(&m_datatype, m_name, prefix))
		return false;

	return true;
}

/*
  Initializing m_params(1) with small preallocation-size, because Parameter
  will be constructed several times for EACH SGObject instance.
 */
Parameter::Parameter() : m_params(1)
{
	SG_REF(sg_io);
}

Parameter::~Parameter()
{
	for (int32_t i = 0; i < get_num_parameters(); i++)
		delete m_params.get_element(i);

	SG_UNREF(sg_io);
}

void Parameter::add_type(
    const TSGDataType* type, void* param, const char* name,
    const char* description)
{
	if (name == NULL || *name == '\0')
		SG_SERROR("FATAL: Parameter::add_type(): `name' is empty!\n")

	for (size_t i = 0; i < strlen(name); ++i)
	{
		if (!std::isalnum(name[i]) && name[i] != '_' && name[i] != '.')
		{
			SG_SERROR(
			    "Character %d of parameter with name \"%s\" is illegal "
			    "(only alnum or underscore is allowed)\n",
			    i, name);
		}
	}

	for (int32_t i = 0; i < get_num_parameters(); i++)
		if (strcmp(m_params.get_element(i)->m_name, name) == 0)
			SG_SERROR(
			    "FATAL: Parameter::add_type(): "
			    "Double parameter `%s'!\n",
			    name);

	m_params.append_element(new TParameter(type, param, name, description));
}

void Parameter::print(const char* prefix)
{
	for (int32_t i = 0; i < get_num_parameters(); i++)
		m_params.get_element(i)->print(prefix);
}

bool Parameter::save(CSerializableFile* file, const char* prefix)
{
	for (int32_t i = 0; i < get_num_parameters(); i++)
	{
		if (!m_params.get_element(i)->save(file, prefix))
			return false;
	}

	return true;
}

bool Parameter::load(CSerializableFile* file, const char* prefix)
{
	for (int32_t i = 0; i < get_num_parameters(); i++)
		if (!m_params.get_element(i)->load(file, prefix))
			return false;

	return true;
}

void Parameter::set_from_parameters(Parameter* params)
{
	/* iterate over parameters in the given list */
	for (index_t i = 0; i < params->get_num_parameters(); ++i)
	{
		TParameter* current = params->get_parameter(i);
		TSGDataType current_type = current->m_datatype;

		ASSERT(m_params.get_num_elements())

		/* search for own parameter with same name and check types if found */
		TParameter* own = NULL;
		for (index_t j = 0; j < m_params.get_num_elements(); ++j)
		{
			own = m_params.get_element(j);
			if (!strcmp(own->m_name, current->m_name))
			{
				if (own->m_datatype == current_type)
				{
					own = m_params.get_element(j);
					break;
				}
				else
				{
					index_t l = 200;
					char* given_type = SG_MALLOC(char, l);
					char* own_type = SG_MALLOC(char, l);
					current->m_datatype.to_string(given_type, l);
					own->m_datatype.to_string(own_type, l);
					SG_SERROR(
					    "given parameter \"%s\" has a different type (%s)"
					    " than existing one (%s)\n",
					    current->m_name, given_type, own_type);
					SG_FREE(given_type);
					SG_FREE(own_type);
				}
			}
			else
				own = NULL;
		}

		if (!own)
		{
			SG_SERROR(
			    "parameter with name %s does not exist\n", current->m_name);
		}

		/* check if parameter contained CSGobjects (update reference counts) */
		if (current_type.m_ptype == PT_SGOBJECT)
		{
			/* PT_SGOBJECT only occurs for ST_NONE */
			if (own->m_datatype.m_stype == ST_NONE)
			{
				if (own->m_datatype.m_ctype == CT_SCALAR)
				{
					CSGObject** to_unref = (CSGObject**)own->m_parameter;
					CSGObject** to_ref = (CSGObject**)current->m_parameter;

					if ((*to_ref) != (*to_unref))
					{
						SG_REF((*to_ref));
						SG_UNREF((*to_unref));
					}
				}
				else
				{
					/* unref all SGObjects and reference the new ones */
					CSGObject*** to_unref = (CSGObject***)own->m_parameter;
					CSGObject*** to_ref = (CSGObject***)current->m_parameter;

					for (index_t j = 0; j < own->m_datatype.get_num_elements();
					     ++j)
					{
						if ((*to_ref)[j] != (*to_unref)[j])
						{
							SG_REF(((*to_ref)[j]));
							SG_UNREF(((*to_unref)[j]));
						}
					}
				}
			}
			else
				SG_SERROR("primitive type PT_SGOBJECT occurred with structure "
				          "type other than ST_NONE");
		}

		/* construct pointers to the to be copied parameter data */
		void* dest = NULL;
		void* source = NULL;
		if (current_type.m_ctype == CT_SCALAR)
		{
			/* for scalar values, just copy content the pointer points to */
			dest = own->m_parameter;
			source = current->m_parameter;

			/* in case of CSGObject, pointers are not equal if CSGObjects are
			 * equal, so check. For other values, the pointers are equal and
			 * the not-copying is handled below before the memcpy call */
			if (own->m_datatype.m_ptype == PT_SGOBJECT)
			{
				if (*((CSGObject**)dest) == *((CSGObject**)source))
				{
					dest = NULL;
					source = NULL;
				}
			}
		}
		else
		{
			/* for matrices and vectors, sadly m_parameter has to be
			 * de-referenced once, because a pointer to the array address is
			 * saved, but the array address itself has to be copied.
			 * consequently, for dereferencing, a type distinction is needed */
			switch (own->m_datatype.m_ptype)
			{
			case PT_FLOAT64:
				dest = *((float64_t**)own->m_parameter);
				source = *((float64_t**)current->m_parameter);
				break;
			case PT_SGOBJECT:
				dest = *((CSGObject**)own->m_parameter);
				source = *((CSGObject**)current->m_parameter);
				break;
			default:
				SG_SNOTIMPLEMENTED
				break;
			}
		}

		/* copy parameter data, size in memory is equal because of same type */
		if (dest != source)
			sg_memcpy(dest, source, own->m_datatype.get_size());
	}
}

void Parameter::add_parameters(Parameter* params)
{
	for (index_t i = 0; i < params->get_num_parameters(); ++i)
	{
		TParameter* current = params->get_parameter(i);
		add_type(
		    &(current->m_datatype), current->m_parameter, current->m_name,
		    current->m_description);
	}
}

bool Parameter::contains_parameter(const char* name)
{
	for (index_t i = 0; i < m_params.get_num_elements(); ++i)
	{
		if (!strcmp(name, m_params[i]->m_name))
			return true;
	}

	return false;
}

bool TParameter::operator==(const TParameter& other) const
{
	bool result = true;
	result &= !strcmp(m_name, other.m_name);
	return result;
}

bool TParameter::operator<(const TParameter& other) const
{
	return strcmp(m_name, other.m_name) < 0;
}

bool TParameter::operator>(const TParameter& other) const
{
	return strcmp(m_name, other.m_name) > 0;
}
