/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Yuyu Zhang, Viktor Gal,
 *          Thoralf Klein, Evan Shelhamer, Sergey Lisitsyn,
 *          Evangelos Anagnostopoulos
 */

#include <shogun/lib/common.h>

#ifndef VERSION_H__
#define VERSION_H__

#include <shogun/lib/config.h>

namespace shogun
{
class RefCount;

/** @brief Class Version provides version information.
 *
 * It provides information of the version of shogun that is currently used, for
 * example the svn revision, time and date of compile and compilation and
 * the linkflags used.
 */
class SHOGUN_EXPORT Version
{
public:
	/** constructor */
	Version();
	/** destructor */
	virtual ~Version();

	/** print version */
	static void print_version();

	/** get main version */
	static const char* get_version_main();

	/** get version extra */
	static const char* get_version_extra();

	/** get version release */
	static const char* get_version_release();

	/** get version revision */
	static int64_t get_version_revision();

	/** get version year */
	static int32_t get_version_year();

	/** get version month */
	static int32_t get_version_month();

	/** get version day */
	static int32_t get_version_day();

	/** get version hour */
	static int32_t get_version_hour();

	/** get version minute */
	static int32_t get_version_minute();

	/** get parameter serialization version */
	static int32_t get_version_parameter();

	/** get version in minutes */
	static int64_t get_version_in_minutes();

	/** ref object
	 * @return ref count
	 */
	int32_t ref();

	/** ref count
	 * @return ref count
	 */
	int32_t ref_count() const;

	/** unref object
	 * @return ref count
	 */
	int32_t unref();

protected:
	/** version release */
	static const char version_release[128];
	/** version main */
	static const char version_main[32];
	/** version extra */
	static const char version_extra[128];
	/** version revision */
	static const int64_t version_revision;
	/** version year */
	static const int32_t version_year;
	/** version month */
	static const int32_t version_month;
	/** version day */
	static const int32_t version_day;
	/** version hour */
	static const int32_t version_hour;
	/** version minute */
	static const int32_t version_minute;
	/** version parameter */
	static const int32_t version_parameter;

private:
	RefCount* m_refcount;
};
}
#endif
