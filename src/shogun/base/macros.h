/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#ifndef __SG_MACROS_H__
#define __SG_MACROS_H__

#if defined(__GNUC__) || defined(__APPLE__)
#define SG_FORCED_INLINE inline __attribute__((always_inline))
#define SG_FORCED_PACKED __attribute__((__packed__))
#define SG_ATTRIBUTE_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
#define SG_FORCED_INLINE __forceinline
#define SG_FORCED_PACKED
#define SG_ATTRIBUTE_UNUSED
#else
#define SG_FORCED_INLINE
#define SG_FORCED_PACKED
#define SG_ATTRIBUTE_UNUSED
#endif

// a quick macro for making sure that an object
// does not have a copy-ctor and operator=
#define SG_DELETE_COPY_AND_ASSIGN(TypeName) \
	TypeName(const TypeName&) = delete; \
	void operator=(const TypeName&) = delete

#endif
