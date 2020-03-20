/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Thoralf Klein, Pan Deng, Evgeniy Andreev,
 *          Viktor Gal, Giovanni De Toni, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/base/ShogunEnv.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/fs/FileSystemRegistry.h>
#include <shogun/io/fs/Path.h>

#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/mathematics/linalg/SGLinalg.h>

#include <rxcpp/rx-lite.hpp>

#include <csignal>
#include <functional>
#include <string>
#include <sstream>

#ifdef HAVE_PROTOBUF
#include <google/protobuf/stubs/common.h>
#endif

using namespace shogun;

ShogunEnv* ShogunEnv::instance()
{
	static ShogunEnv shogun_env{};
	return &shogun_env;
}

ShogunEnv::ShogunEnv()
{
#ifndef _MSC_VER
	sg_io = std::make_unique<io::SGIO>();
#endif
	sg_linalg = std::make_unique<SGLinalg>();
	sg_signal = std::make_unique<Signal>();

	sg_fequals_epsilon = 0.0;
	sg_fequals_tolerant = false;

	// Set up signal handler
	std::signal(SIGINT, sg_signal->handler);
	init_from_env();
}

ShogunEnv::~ShogunEnv()
{
	delete Signal::m_subscriber;
	delete Signal::m_observable;
	delete Signal::m_subject;

#ifdef HAVE_PROTOBUF
	::google::protobuf::ShutdownProtobufLibrary();
#endif
}

void ShogunEnv::init_from_env()
{
	char* env_log_val = NULL;
	env_log_val = getenv("SHOGUN_LOG_LEVEL");
	if (env_log_val && sg_io)
	{
		if (strncmp(env_log_val, "TRACE", 5) == 0)
			sg_io->set_loglevel(io::MSG_TRACE);
		else if (strncmp(env_log_val, "DEBUG", 5) == 0)
			sg_io->set_loglevel(io::MSG_DEBUG);
		else if (strncmp(env_log_val, "INFO", 4) == 0)
			sg_io->set_loglevel(io::MSG_INFO);
		else if (strncmp(env_log_val, "WARN", 4) == 0)
			sg_io->set_loglevel(io::MSG_WARN);
		else if (strncmp(env_log_val, "ERROR", 5) == 0)
			sg_io->set_loglevel(io::MSG_ERROR);
	}

	char* env_warnings_val = NULL;
	env_warnings_val = getenv("SHOGUN_GPU_WARNINGS");
	if (env_warnings_val)
	{
		if (strncmp(env_warnings_val, "off", 3) == 0)
			sg_linalg->set_linalg_warnings(false);
	}

	char* env_thread_val = NULL;
	env_thread_val = getenv("SHOGUN_NUM_THREADS");
	if (env_thread_val)
	{
		try
		{
			set_num_threads(std::stoi(std::string(env_thread_val)));
		}
		catch (...)
		{
			sg_io->message(
			    io::MSG_WARN,
			    "The specified SHOGUN_NUM_THREADS environment ({})"
			    "variable could not be parsed as integer!\n",
			    env_thread_val);
		}
	}
}

io::SGIO* ShogunEnv::io()
{
#ifdef _MSC_VER
	if (SG_UNLIKELY(!sg_io))
	{
		sg_io = std::make_unique<io::SGIO>();
		init_from_env();
	}
#endif
	return sg_io.get();
}

float64_t ShogunEnv::fequals_epsilon()
{
	return sg_fequals_epsilon;
}

void ShogunEnv::set_global_fequals_epsilon(float64_t fequals_epsilon)
{
	sg_fequals_epsilon = fequals_epsilon;
}

void ShogunEnv::set_global_fequals_tolerant(bool fequals_tolerant)
{
	sg_fequals_tolerant = fequals_tolerant;
}

bool ShogunEnv::fequals_tolerant()
{
	return sg_fequals_tolerant;
}

Signal* ShogunEnv::signal()
{
	return sg_signal.get();
}

SGLinalg* ShogunEnv::linalg()
{
	return sg_linalg.get();
}

static bool is_shared_lib_name(std::string_view filename)
{
#if defined(__APPLE__)
	static constexpr std::string_view kSharedLibSuffix = ".dylib";
#elif defined(_WIN32)
	static constexpr std::string_view kSharedLibSuffix = ".dll";
#else
	static constexpr std::string_view kSharedLibSuffix = ".so";
#endif

	if (filename.length() > kSharedLibSuffix.size())
	{
#if defined(__APPLE__) || defined(_WIN32)
		auto suffix = filename.substr(filename.length()-kSharedLibSuffix.size(), kSharedLibSuffix.size());
		if (suffix == kSharedLibSuffix)
			return true;
#else
		// UNIX shared lib names
		// libname.so or libname.so.1.1.1
		if (filename.find(kSharedLibSuffix) != std::string::npos)
			return true;
#endif
	}
	return false;
}


std::vector<std::string> ShogunEnv::plugins() const
{
#if defined(_WIN32)
	static constexpr char kPathSeparator= ';';
#else
	static constexpr char kPathSeparator = ':';
#endif

	std::vector<std::string> plugins;

	char* env_plugins_path = NULL;
	env_plugins_path = getenv("SHOGUN_PLUGINS_PATH");
	if (env_plugins_path)
	{
		sg_io->message(io::MSG_TRACE, {},
			"SHOGUN_PLUGINS_PATH environment variable is set to {}",
			env_plugins_path);

		std::istringstream plugins_path(env_plugins_path);
		std::string path;
		while (getline(plugins_path, path, kPathSeparator))
		{
			auto r = is_directory(path);
			if (r)
			{
				io::warn("{} is not a directory: {}",
					path, r.message());
				continue;
			}

			sg_io->message(io::MSG_DEBUG, {},
				"Using {} path as plugin directory", path.c_str());
			std::vector<std::string> libraries;
			r = get_children(path, &libraries);
			if (r)
			{
				io::warn("Could not list files in {}", path);
				continue;
			}

			for (const auto& v: libraries) {
				if (is_shared_lib_name(v))
					plugins.push_back(io::join_path(path, v));
			}
		}
	}
	return plugins;
}
