#include <gtest/gtest.h>
#include <shogun/io/SGIO.h>
#include <stdexcept>

#include <spdlog/sinks/base_sink.h>
#include <spdlog/version.h>

#include <mutex>

using namespace shogun;
using namespace shogun::io;

TEST(SGIO, exception)
{
	EXPECT_THROW(error("Error"), ShogunException);
	EXPECT_THROW(error<std::invalid_argument>("Error"), std::invalid_argument);
	EXPECT_THROW(require(0, "Error"), ShogunException);
	EXPECT_THROW(
	    require<std::invalid_argument>(0, "Error"), std::invalid_argument);
}

// same as spdlog's ostream_sink, but with public message counter
class StreamSink : public spdlog::sinks::base_sink<std::mutex>
{
public:
	explicit StreamSink(std::ostream& os) : counter(0), ostream_(os)
	{
	}

	std::atomic<int32_t> counter;

protected:
	void sink_it_(const spdlog::details::log_msg& msg) override
	{
#if ((SPDLOG_VER_MAJOR == 1) && (SPDLOG_VER_MINOR < 4))
		fmt::memory_buffer formatted;
		sink::formatter_->format(msg, formatted);
#else
		spdlog::memory_buf_t formatted;
		formatter_->format(msg, formatted);
#endif
		ostream_.write(
		    formatted.data(), static_cast<std::streamsize>(formatted.size()));
		ostream_.flush();
		counter++;
	}

	void flush_() override
	{
		ostream_.flush();
	}

	std::ostream& ostream_;
};

bool ends_with(const std::string& long_str, const std::string& short_str)
{
	return long_str.compare(
	           long_str.length() - short_str.length(), short_str.length(),
	           short_str) == 0 ||
	       long_str.compare(
	           long_str.length() - short_str.length() - 1, short_str.length(),
	           short_str) == 0;
}

TEST(SGIO, custom_stderr_sink)
{
	const std::string stderr_log = "Testing that \'error\' writes to stderr.";

	std::ostringstream stderr_stream;
	auto stderr_sink = std::make_shared<StreamSink>(stderr_stream);
	env()->io()->redirect_stderr(stderr_sink);

	EXPECT_THROW(error("{}", stderr_log), ShogunException);

	// busy waiting since it's an async logger (probably not the best way)
	while (stderr_sink->counter.load(std::memory_order_relaxed) != 1)
		;

	std::string stderr_str = stderr_stream.str();

	// check that stderr_str ends with stderr_log
	EXPECT_TRUE(ends_with(stderr_str, stderr_log));

	env()->io()->init_default_sink();
}

TEST(SGIO, custom_stdout_sink)
{
	const std::string stdout_log1 = "Testing unformatted ";
	const std::string stdout_log2 = "print function with custom stdout sink.\n";

	std::ostringstream stdout_stream;
	auto stdout_sink = std::make_shared<StreamSink>(stdout_stream);
	env()->io()->redirect_stdout(stdout_sink);

	print("{}", stdout_log1);
	print("{}", stdout_log2);

	// busy waiting since it's an async logger (probably not the best way)
	while (stdout_sink->counter.load(std::memory_order_relaxed) != 2)
		;

	std::string stdout_str = stdout_stream.str();

	EXPECT_EQ(stdout_stream.str(), stdout_log1 + stdout_log2);

	env()->io()->init_default_sink();
}

TEST(SGIO, loglevels_redirection)
{
	const std::string test_msg = "TEST\n";

	std::ostringstream stdout_stream;
	std::ostringstream stderr_stream;
	auto stdout_sink = std::make_shared<StreamSink>(stdout_stream);
	auto stderr_sink = std::make_shared<StreamSink>(stderr_stream);
	env()->io()->redirect_stdout(stdout_sink);
	env()->io()->redirect_stderr(stderr_sink);

	EMessageType loglevels[] = {MSG_TRACE,    MSG_DEBUG, MSG_INFO,
	                            MSG_WARN,       MSG_ERROR, MSG_CRITICAL,
	                            MSG_MESSAGEONLY};
	env()->io()->set_loglevel(loglevels[0]);

	for (EMessageType loglevel : loglevels)
	{
		env()->io()->message(loglevel, {}, "{}", test_msg);
		while (stdout_sink->counter.load(std::memory_order_relaxed) == 0 &&
		       stderr_sink->counter.load(std::memory_order_relaxed) == 0)
			;

		if (loglevel != MSG_ERROR)
		{
			EXPECT_EQ(stdout_sink->counter.load(std::memory_order_relaxed), 1);
			EXPECT_TRUE(ends_with(stdout_stream.str(), test_msg));
		}
		else
		{
			EXPECT_EQ(stderr_sink->counter.load(std::memory_order_relaxed), 1);
			EXPECT_TRUE(ends_with(stderr_stream.str(), test_msg));
		}

		stdout_sink->counter = 0;
		stderr_sink->counter = 0;
	}

	env()->io()->init_default_sink();
}