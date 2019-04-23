#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <shogun/base/init.h>
#include <shogun/io/SGIO.h>

#include "environments/LinearTestEnvironment.h"
#include "environments/MultiLabelTestEnvironment.h"
#include "environments/RegressionTestEnvironment.h"

using namespace shogun;
using ::testing::Test;
using ::testing::UnitTest;
using ::testing::TestCase;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::TestEventListener;
using ::testing::Environment;

class FailurePrinter : public TestEventListener {
public:
	explicit FailurePrinter(TestEventListener* listener) : TestEventListener() {_listener = listener;}

	virtual ~FailurePrinter() {}

	virtual void OnTestProgramStart(const UnitTest& unit_test) {}
	virtual void OnTestIterationStart(const UnitTest& unit_test, int iteration) {}
	virtual void OnEnvironmentsSetUpStart(const UnitTest& unit_test) {}
	virtual void OnEnvironmentsSetUpEnd(const UnitTest& unit_test) {}
	virtual void OnTestCaseStart(const TestCase& test_case) {}
	virtual void OnTestStart(const TestInfo& test_info) {}
	virtual void OnTestPartResult(const TestPartResult& result);
	virtual void OnTestEnd(const TestInfo& test_info);
	virtual void OnTestCaseEnd(const TestCase& test_case) {}
	virtual void OnEnvironmentsTearDownStart(const UnitTest& unit_test) { }
	virtual void OnEnvironmentsTearDownEnd(const UnitTest& unit_test) { }
	virtual void OnTestIterationEnd(const UnitTest& unit_test, int iteration) { _listener->OnTestIterationEnd(unit_test, iteration); }
	virtual void OnTestProgramEnd(const UnitTest& unit_test) { }

protected:
	TestEventListener* _listener;
};

void FailurePrinter::OnTestPartResult(const TestPartResult& test_part_result)
{
	if (test_part_result.failed())
	{
		_listener->OnTestPartResult(test_part_result);
		printf("\n");
	}
}

void FailurePrinter::OnTestEnd(const TestInfo& test_info)
{
	if (test_info.result()->Failed())
		_listener->OnTestEnd(test_info);
}

LinearTestEnvironment* linear_test_env;
MultiLabelTestEnvironment* multilabel_test_env;
RegressionTestEnvironment* regression_test_env;

int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	::testing::InitGoogleMock(&argc, argv);

	if (argc > 1 && !strcmp(argv[1], "--only-on-failure"))
	{
		testing::TestEventListeners& listeners =
			testing::UnitTest::GetInstance()->listeners();

		testing::TestEventListener* default_printer
			= listeners.Release(listeners.default_result_printer());
		listeners.Append(new FailurePrinter(default_printer));
	}

	linear_test_env = new LinearTestEnvironment();
	::testing::AddGlobalTestEnvironment(linear_test_env);

	multilabel_test_env = new MultiLabelTestEnvironment();
	::testing::AddGlobalTestEnvironment(multilabel_test_env);

	regression_test_env = new RegressionTestEnvironment();
	::testing::AddGlobalTestEnvironment(regression_test_env);

	init_shogun_with_defaults();

	int ret = RUN_ALL_TESTS();

	exit_shogun();

	return ret;
}

