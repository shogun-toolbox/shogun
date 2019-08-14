/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Giovanni De Toni, Viktor Gal, Heiko Strathmann,
 *          Thoralf Klein, Evangelos Anagnostopoulos, Weijie Lin, Bjoern Esser,
 *          Saurabh Goyal
 */

#define SPDLOG_EOL ""

#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/async_logger.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <mutex>

using namespace shogun;
using namespace shogun::io;

class Formatter : public spdlog::formatter
{
public:
	Formatter(const Formatter& orig)
	    : formatter_(orig.formatter_->clone())
	{
	}

	template <typename... Args>
	Formatter(Args&&... args)
	    : formatter_(std::make_unique<spdlog::pattern_formatter>(args...))
	{
	}

	void format(
	    const spdlog::details::log_msg& msg, fmt::memory_buffer& dest) override
	{
		if (msg.level == static_cast<spdlog::level::level_enum>(MSG_MESSAGEONLY))
			dest.append(msg.payload.data(), msg.payload.data() + msg.payload.size());
		else
			formatter_->format(msg, dest);
	}

	std::unique_ptr<spdlog::formatter> clone() const override
	{
		return std::make_unique<Formatter>(*this);
	}

private:
	std::unique_ptr<spdlog::formatter> formatter_;
};

class SGIO::RedirectSink : public spdlog::sinks::base_sink<std::mutex>
{
public:
	RedirectSink()
	{
		stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
		stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
	}

	void redirect_stdout(std::shared_ptr<spdlog::sinks::sink> sink_)
	{
		set_sink_(stdout_sink, sink_);
	}

	void redirect_stderr(std::shared_ptr<spdlog::sinks::sink> sink_)
	{
		set_sink_(stderr_sink, sink_);
	}

protected:
	void sink_it_(const spdlog::details::log_msg& msg) override
	{
		if (msg.level == spdlog::level::err)
			stderr_sink->log(msg);
		else
			stdout_sink->log(msg);
	}

	void flush_() override
	{
		stdout_sink->flush();
		if (stdout_sink != stderr_sink)
			stderr_sink->flush();
	}

	void set_pattern_(const std::string& pattern) override
	{
		set_formatter_(std::make_unique<Formatter>(pattern));
	}

	void
	set_formatter_(std::unique_ptr<spdlog::formatter> sink_formatter) override
	{
		formatter_ = std::move(sink_formatter);
		stdout_sink->set_formatter(formatter_->clone());
		if (stdout_sink != stderr_sink)
			stderr_sink->set_formatter(formatter_->clone());
	}

private:
	void set_sink_(
	    std::shared_ptr<spdlog::sinks::sink>& old_sink,
	    const std::shared_ptr<spdlog::sinks::sink>& new_sink)
	{
		flush();
		std::lock_guard<std::mutex> lock(mutex_);
		if (new_sink)
			old_sink = new_sink;
		else
			old_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
	}

	std::shared_ptr<spdlog::sinks::sink> stdout_sink;
	std::shared_ptr<spdlog::sinks::sink> stderr_sink;
};

SGIO::SGIO()
    : show_progress(false), syntax_highlight(true)
{
	init_default_sink();
	init_default_logger();
	update_pattern();
}

void SGIO::init_default_logger(uint64_t queue_size, uint64_t n_threads)
{
	thread_pool =
	    std::make_shared<spdlog::details::thread_pool>(queue_size, n_threads);
	io_logger = std::make_shared<spdlog::async_logger>(
	    "sg_global", io_sink, thread_pool, spdlog::async_overflow_policy::block);
}

void SGIO::init_default_sink()
{
	io_sink = std::make_shared<RedirectSink>();
	if (io_logger)
	{
		io_logger->sinks().clear();
		io_logger->sinks().push_back(io_sink);
	}
}

void SGIO::message_(
    EMessageType prio, const SourceLocation& loc,
    const fmt::string_view& msg) const
{
	io_logger->log(
	    {loc.file, loc.line, loc.function},
	    static_cast<spdlog::level::level_enum>(prio), msg);
}

EMessageType SGIO::get_loglevel() const
{
	return static_cast<EMessageType>(io_logger->level());
}

void SGIO::set_loglevel(EMessageType level)
{
	io_logger->set_level(static_cast<spdlog::level::level_enum>(level));
}

char* SGIO::c_string_of_substring(substring s)
{
	uint32_t len = s.end - s.start + 1;
	char* ret = SG_CALLOC(char, len);
	sg_memcpy(ret, s.start, len - 1);
	return ret;
}

float32_t SGIO::float_of_substring(substring s)
{
	char* endptr = s.end;
	float32_t f = strtof(s.start, &endptr);
	if (endptr == s.start && s.start != s.end)
		error("{} is not a float!\n", c_string_of_substring(s));

	return f;
}

float64_t SGIO::double_of_substring(substring s)
{
	char* endptr = s.end;
	float64_t f = strtod(s.start, &endptr);
	if (endptr == s.start && s.start != s.end)
		error("{} is not a double!\n", c_string_of_substring(s));

	return f;
}

int32_t SGIO::int_of_substring(substring s)
{
	char* c_string = c_string_of_substring(s);
	int32_t int_val = atoi(c_string);
	SG_FREE(c_string);

	return int_val;
}

uint32_t SGIO::ulong_of_substring(substring s)
{
	return strtoul(s.start, NULL, 10);
}

uint32_t SGIO::ss_length(substring s)
{
	return (s.end - s.start);
}

SGIO::~SGIO()
{
	io_logger->flush();
}

void SGIO::update_pattern()
{
	std::stringstream pattern_builder;
	pattern_builder << "[%D %T %! %@";

	if (syntax_highlight)
		pattern_builder << "%^%l%$] ";
	else
		pattern_builder << "%l] ";

	pattern_builder << "%v";

	io_logger->set_formatter(std::make_unique<Formatter>(pattern_builder.str()));
}

void SGIO::redirect_stdout(std::shared_ptr<spdlog::sinks::sink> sink)
{
	io_sink->redirect_stdout(sink);
	update_pattern();
}

void SGIO::redirect_stderr(std::shared_ptr<spdlog::sinks::sink> sink)
{
	io_sink->redirect_stderr(sink);
	update_pattern();
}

bool SGIO::should_log(EMessageType prio) const
{
	return io_logger->should_log(static_cast<spdlog::level::level_enum>(prio));
}
