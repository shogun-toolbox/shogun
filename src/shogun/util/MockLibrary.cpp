#include "shogun/util/MockBaseClass.h"

/**
 * This library is a mock-library used to test
 * the plugin framework in Plugin_unittest.cc.
 */
namespace shogun
{
    class MockClass : public MockBaseClass
    {
        const char* get_name() const override
        {
            return "MockClass";
        }
    };

    class AnotherMockClass : public MockBaseClass
    {
        const char* get_name() const override
        {
            return "AnotherMockClass";
        }
    };

    template<typename T>
    class TemplatedClass : public MockBaseClass
    {
        const char* get_name() const override
        {
            return "TemplatedClass";
        }
    };

    BEGIN_MANIFEST("Mock library")
    EXPORT(MockClass, MockBaseClass, "mock_class")
    EXPORT(AnotherMockClass, MockBaseClass, "another_mock_class")
    //EXPORT(TemplatedClass, MockBaseClass, "templated_mock_class")
    END_MANIFEST()
}
