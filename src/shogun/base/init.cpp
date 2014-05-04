/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/lib/memory.h>
#include <shogun/lib/config.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Random.h>
#include <shogun/mathematics/linalg/global/LinearAlgebra.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parallel.h>
#include <shogun/base/Version.h>
#include <shogun/base/SGRefObject.h>
#include <stdlib.h>
#include <string.h>

#ifdef TRACE_MEMORY_ALLOCS
#include <shogun/lib/Map.h>
shogun::CMap<void*, shogun::MemoryBlock>* sg_mallocs=NULL;
#endif

#ifdef HAVE_PROTOBUF
#include <google/protobuf/stubs/common.h>
#endif

namespace shogun
{
	Parallel* sg_parallel=NULL;
	SGIO* sg_io=NULL;
	Version* sg_version=NULL;
	CMath* sg_math=NULL;
	CRandom* sg_rand=NULL;
	CLinearAlgebra* sg_linalg=NULL;

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
		if (!sg_linalg)
			sg_linalg = new shogun::CLinearAlgebra();
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
		SG_REF(sg_linalg);		

		sg_print_message=print_message;
		sg_print_warning=print_warning;
		sg_print_error=print_error;
		sg_cancel_computations=cancel_computations;
		
		init_from_env();
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
		SG_UNREF(sg_linalg);

#ifdef HAVE_PROTOBUF
		::google::protobuf::ShutdownProtobufLibrary();
#endif
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

	void set_global_linalg(CLinearAlgebra* linalg)
	{
		SG_REF(linalg);
		SG_UNREF(sg_linalg);
		sg_linalg=linalg;
	}

	CLinearAlgebra* get_global_linalg()
	{
		SG_REF(sg_linalg);
		return sg_linalg;
	}

	void init_from_env()
	{
		char* env_log_val = NULL;
		SGIO* io = get_global_io();
		env_log_val = getenv("SHOGUN_LOG_LEVEL");
		if (env_log_val)
		{
			if(strncmp(env_log_val, "DEBUG", 5) == 0)
				io->set_loglevel(MSG_DEBUG);
			else if(strncmp(env_log_val, "WARN", 4) == 0)
				io->set_loglevel(MSG_WARN);
			else if(strncmp(env_log_val, "ERROR", 5) == 0)
				io->set_loglevel(MSG_ERROR);
		}
		SG_UNREF(io);
	}
}
