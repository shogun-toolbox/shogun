/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Thoralf Klein, Pan Deng, Evgeniy Andreev,
 *          Viktor Gal, Giovanni De Toni, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/base/init.h>
#include <shogun/lib/memory.h>
#include <shogun/lib/config.h>

#include <shogun/base/Parallel.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Version.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>

#include <rxcpp/rx-lite.hpp>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/SGLinalg.h>

#include <csignal>
#include <functional>
#include <stdlib.h>
#include <string.h>
#include <string>

#ifdef HAVE_PROTOBUF
#include <google/protobuf/stubs/common.h>
#endif

namespace shogun
{
	std::shared_ptr<Parallel> sg_parallel=NULL;
	std::shared_ptr<SGIO> sg_io=NULL;
	std::shared_ptr<Version> sg_version=NULL;
	std::unique_ptr<Signal> sg_signal(nullptr);
	std::unique_ptr<SGLinalg> sg_linalg(nullptr);

	// Two global variables to over-ride Math::fequals for certain
	// serialization
	// unit tests to pass. These should be removed if possible and serialization
	// formats should be fixed.
	float64_t sg_fequals_epsilon = 0.0;
	bool sg_fequals_tolerant = 0.0;

	/// function called to print normal messages
	std::function<void(FILE*, const char*)> sg_print_message(nullptr);

	/// function called to print warning messages
	std::function<void(FILE*, const char*)> sg_print_warning(nullptr);

	/// function called to print error messages
	std::function<void(FILE*, const char*)> sg_print_error(nullptr);

	void init_shogun(
	    const std::function<void(FILE*, const char*)> print_message,
	    const std::function<void(FILE*, const char*)> print_warning,
	    const std::function<void(FILE*, const char*)> print_error)
	{
		if (!sg_io)
			sg_io = std::make_shared<shogun::SGIO>();
		if (!sg_parallel)
			sg_parallel=std::make_shared<shogun::Parallel>();
		if (!sg_version)
			sg_version = std::make_shared<shogun::Version>();
		if (!sg_linalg)
			sg_linalg = std::unique_ptr<SGLinalg>(new shogun::SGLinalg());
		if (!sg_signal)
			sg_signal = std::unique_ptr<Signal>(new shogun::Signal());

		sg_print_message=print_message;
		sg_print_warning=print_warning;
		sg_print_error=print_error;

		// Set up signal handler
		std::signal(SIGINT, sg_signal->handler);

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
		sg_version.reset();
		sg_parallel.reset();
		sg_io.reset();

		delete Signal::m_subscriber;
		delete Signal::m_observable;
		delete Signal::m_subject;

#ifdef HAVE_PROTOBUF
		::google::protobuf::ShutdownProtobufLibrary();
#endif
	}

	void set_global_io(std::shared_ptr<SGIO> io)
	{
		sg_io=io;
	}

	std::shared_ptr<SGIO> get_global_io()
	{
		return sg_io;
	}

	float64_t get_global_fequals_epsilon()
	{
		return sg_fequals_epsilon;
	}

	void set_global_fequals_epsilon(float64_t fequals_epsilon)
	{
		sg_fequals_epsilon = fequals_epsilon;
	}

	void set_global_fequals_tolerant(bool fequals_tolerant)
	{
		sg_fequals_tolerant = fequals_tolerant;
	}

	bool get_global_fequals_tolerant()
	{
		return sg_fequals_tolerant;
	}

	void set_global_parallel(std::shared_ptr<Parallel> parallel)
	{
		sg_parallel=parallel;
	}

	std::shared_ptr<Parallel> get_global_parallel()
	{
		return sg_parallel;
	}

	void set_global_version(std::shared_ptr<Version> version)
	{
		sg_version=version;
	}

	std::shared_ptr<Version> get_global_version()
	{
		return sg_version;
	}

	Signal* get_global_signal()
	{
		return sg_signal.get();
	}

#ifndef SWIG // SWIG should skip this part
	SGLinalg* get_global_linalg()
	{
		return sg_linalg.get();
	}
#endif

	void init_from_env()
	{
		char* env_log_val = NULL;
		auto io = get_global_io();
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

		char* env_warnings_val = NULL;
		auto linalg = get_global_linalg();
		env_warnings_val = getenv("SHOGUN_GPU_WARNINGS");
		if (env_warnings_val)
		{
			if (strncmp(env_warnings_val, "off", 3) == 0)
				linalg->set_linalg_warnings(false);
		}

		char* env_thread_val = NULL;
		auto parallel = get_global_parallel();
		env_thread_val = getenv("SHOGUN_NUM_THREADS");
		if (env_thread_val)
		{

			try {
				int32_t num_threads = std::stoi(std::string(env_thread_val));
				parallel->set_num_threads(num_threads);
			} catch (...) {
				SG_WARNING("The specified SHOGUN_NUM_THREADS environment (%s)"
				"variable could not be parsed as integer!\n", env_thread_val);
			}
		}
	}
}
