/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <base/init.h>
#include <mathematics/Math.h>
#include <mathematics/Random.h>
#include <lib/common.h>
#include <lib/Map.h>
#include <base/Parallel.h>
#include <base/Version.h>

#ifdef TRACE_MEMORY_ALLOCS
shogun::CMap<void*, shogun::MemoryBlock>* sg_mallocs=NULL;
#endif

namespace shogun
{
	Parallel* sg_parallel=NULL;
	SGIO* sg_io=NULL;
	Version* sg_version=NULL;
	CMath* sg_math=NULL;
	CRandom* sg_rand=NULL;

	/// function called to print normal messages
	void (*sg_print_message)(FILE* target, const char* str) = NULL;

	/// function called to print warning messages
	void (*sg_print_warning)(FILE* target, const char* str) = NULL;

	/// function called to print error messages
	void (*sg_print_error)(FILE* target, const char* str) = NULL;

	/// function called to cancel things
	void (*sg_cancel_computations)(bool &delayed, bool &immediately)=NULL;


	void init_shogun(void (*print_message)(FILE* target, const char* str),
			void (*print_warning)(FILE* target, const char* str),
			void (*print_error)(FILE* target, const char* str),
			void (*cancel_computations)(bool &delayed, bool &immediately))
	{
		if (!sg_io)
			sg_io = new shogun::SGIO();
		if (!sg_parallel)
			sg_parallel=new shogun::Parallel();
		if (!sg_version)
			sg_version = new shogun::Version();
		if (!sg_math)
			sg_math = new shogun::CMath();
		if (!sg_rand)
			sg_rand = new shogun::CRandom();
#ifdef TRACE_MEMORY_ALLOCS
		if (!sg_mallocs)
			sg_mallocs = new shogun::CMap<void*, MemoryBlock>(631, 1024, false);

		SG_REF(sg_mallocs);
#endif
		SG_REF(sg_io);
		SG_REF(sg_parallel);
		SG_REF(sg_version);
		SG_REF(sg_math);
		SG_REF(sg_rand);

		sg_print_message=print_message;
		sg_print_warning=print_warning;
		sg_print_error=print_error;
		sg_cancel_computations=cancel_computations;
	}

	void sg_global_print_default(FILE* target, const char* str)
	{
		fprintf(target, "%s", str);
	}

	void init_shogun_with_defaults()
	{
		init_shogun(&sg_global_print_default, &sg_global_print_default,
				&sg_global_print_default);
	}

	void exit_shogun()
	{
#ifdef TRACE_MEMORY_ALLOCS
		list_memory_allocs();
		shogun::CMap<void*, shogun::MemoryBlock>* mallocs=sg_mallocs;
		sg_mallocs=NULL;
		SG_UNREF(mallocs);
#endif
		sg_print_message=NULL;
		sg_print_warning=NULL;
		sg_print_error=NULL;
		sg_cancel_computations=NULL;

		SG_UNREF(sg_rand);
		SG_UNREF(sg_math);
		SG_UNREF(sg_version);
		SG_UNREF(sg_parallel);
		SG_UNREF(sg_io);

	}

	void set_global_io(SGIO* io)
	{
		SG_REF(io);
		SG_UNREF(sg_io);
		sg_io=io;
	}

	SGIO* get_global_io()
	{
		SG_REF(sg_io);
		return sg_io;
	}

	void set_global_parallel(Parallel* parallel)
	{
		SG_REF(parallel);
		SG_UNREF(sg_parallel);
		sg_parallel=parallel;
	}

	Parallel* get_global_parallel()
	{
		SG_REF(sg_parallel);
		return sg_parallel;
	}

	void set_global_version(Version* version)
	{
		SG_REF(version);
		SG_UNREF(sg_version);
		sg_version=version;
	}

	Version* get_global_version()
	{
		SG_REF(sg_version);
		return sg_version;
	}

	void set_global_math(CMath* math)
	{
		SG_REF(math);
		SG_UNREF(sg_math);
		sg_math=math;
	}

	CMath* get_global_math()
	{
		SG_REF(sg_math);
		return sg_math;
	}

	void set_global_rand(CRandom* rand)
	{
		SG_REF(rand);
		SG_UNREF(sg_rand);
		sg_rand=rand;
	}

	CRandom* get_global_rand()
	{
		SG_REF(sg_rand);
		return sg_rand;
	}
}
