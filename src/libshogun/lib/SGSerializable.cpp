/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */

#include "SGSerializable.h"
#include "lib/Parameter.h"

namespace shogun {

template<> void
CSGSerializable::set_generic<bool>(void)
{ m_generic = PT_BOOL; }

template<> void
CSGSerializable::set_generic<char>(void)
{ m_generic = PT_CHAR; }

template<> void
CSGSerializable::set_generic<int8_t>(void)
{ m_generic = PT_INT8; }

template<> void
CSGSerializable::set_generic<uint8_t>(void)
{ m_generic = PT_UINT8; }

template<> void
CSGSerializable::set_generic<int16_t>(void)
{ m_generic = PT_INT16; }

template<> void
CSGSerializable::set_generic<uint16_t>(void)
{ m_generic = PT_UINT16; }

template<> void
CSGSerializable::set_generic<int32_t>(void)
{ m_generic = PT_INT32; }

template<> void
CSGSerializable::set_generic<uint32_t>(void)
{ m_generic = PT_UINT32; }

template<> void
CSGSerializable::set_generic<int64_t>(void)
{ m_generic = PT_INT64; }

template<> void
CSGSerializable::set_generic<uint64_t>(void)
{ m_generic = PT_UINT64; }

template<> void
CSGSerializable::set_generic<float32_t>(void)
{ m_generic = PT_FLOAT32; }

template<> void
CSGSerializable::set_generic<float64_t>(void)
{ m_generic = PT_FLOAT64; }

template<> void
CSGSerializable::set_generic<floatmax_t>(void)
{ m_generic = PT_FLOATMAX; }

} /* namespace shogun  */


using namespace shogun;

extern IO* sg_io;

CSGSerializable::CSGSerializable(void)
{
	m_parameters = new Parameter(sg_io);
	m_generic = PT_NOT_GENERIC;
}

CSGSerializable::~CSGSerializable(void)
{
	delete m_parameters;
}

bool
CSGSerializable::is_generic(EPrimitveType* generic) const
{
	*generic = m_generic;

	return m_generic != PT_NOT_GENERIC;
}

void
CSGSerializable::unset_generic(void)
{ m_generic = PT_NOT_GENERIC; }

void
CSGSerializable::print_serializable(const char* prefix)
{
	m_parameters->print(prefix);
}

bool
CSGSerializable::save_serializable(CSerializableFile* file,
								   const char* prefix)
{
	bool result = m_parameters->save(file, prefix);

	if (prefix == NULL || *prefix == '\0')
		file->close();

	return result;
}

bool
CSGSerializable::load_serializable(CSerializableFile* file,
								   const char* prefix)
{
	bool result = m_parameters->load(file, prefix);

	if (prefix == NULL || *prefix == '\0')
		file->close();

	return result;
}
