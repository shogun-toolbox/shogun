#include <../tests/unit/base/MockBaseClass.h>

/**
 * This library is a mock-library used to test
 * the plugin framework in Plugin_unittest.cc.
 */
namespace shogun
{
    class MockClass : public MockBaseClass
    {
        virtual const char* get_name() const
        {
            return "MockClass";
        }
    };

    class AnotherMockClass : public MockBaseClass
    {
        virtual const char* get_name() const
        {
            return "AnotherMockClass";
        }
    };

    BEGIN_MANIFEST("Mock library")
    EXPORT(MockClass, MockBaseClass, "mock_class")
    EXPORT(AnotherMockClass, MockBaseClass, "another_mock_class")
    END_MANIFEST()
}
