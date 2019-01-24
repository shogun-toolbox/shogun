/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Viktor Gal, Heiko Strathmann, Thoralf Klein, 
 *          Evan Shelhamer, Bjoern Esser, Evangelos Anagnostopoulos
 */

#include <shogun/base/Version.h>
#include <shogun/lib/versionstring.h>
#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/RefCount.h>

using namespace shogun;

namespace shogun
{
// TODO: in C++17 inline all of this in header file and remove it from here
constexpr int64_t Version::version_revision;
constexpr int32_t Version::version_year;
constexpr int32_t Version::version_month;
constexpr int32_t Version::version_day;
constexpr int32_t Version::version_hour;
constexpr int32_t Version::version_minute;
constexpr int32_t Version::version_parameter;
constexpr const char Version::version_extra[128];
constexpr const char Version::version_release[128];
constexpr const char Version::version_main[32];
}

Version::Version()
{
	m_refcount = new RefCount();
}


Version::~Version()
{
	delete m_refcount;
}

void Version::print_version()
{
	SG_SPRINT("libshogun (%s/%s%" PRId64 ")\n\n", MACHINE, Version::get_version_release(), Version::get_version_revision())
	SG_SPRINT("Copyright (C) 1999-2009 Fraunhofer Institute FIRST\n")
	SG_SPRINT("Copyright (C) 1999-2011 Max Planck Society\n")
	SG_SPRINT("Copyright (C) 2009-2011 Berlin Institute of Technology\n")
	SG_SPRINT("Copyright (C) 2012-2014 Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann, Viktor Gal, Fernando Iglesias et al\n")
	SG_SPRINT("Written   (W) 1999-2012 Soeren Sonnenburg, Gunnar Raetsch et al.\n\n")
#ifndef USE_SVMLIGHT
	SG_SPRINT("This is free software; see the source for copying conditions.  There is NO\n")
	SG_SPRINT("warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\n")
#endif
}

int32_t Version::ref()
{
	return m_refcount->ref();
}

int32_t Version::ref_count() const
{
	return m_refcount->ref_count();
}

int32_t Version::unref()
{
	int32_t rc = m_refcount->unref();

	if (rc==0)
	{
		delete this;
		return 0;
	}

	return rc;
}
