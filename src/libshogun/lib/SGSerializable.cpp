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

using namespace shogun;

extern IO* sg_io;

CSGSerializable::CSGSerializable(void)
{
	m_parameters = new Parameter(sg_io);
}

CSGSerializable::~CSGSerializable(void)
{
	delete m_parameters;
}

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
