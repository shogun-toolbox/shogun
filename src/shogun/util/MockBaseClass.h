#ifndef _MOCK_BASE_CLASS_H_
#define _MOCK_BASE_CLASS_H_

#include <shogun/base/SGObject.h>
#include <shogun/base/library.h>

namespace shogun
{
    /** @brief
     * This class is used by Plugin unit-tests.
     */
    class MockBaseClass : public SGObject
    {
    public:
        const char* get_name() const override
        {
            return "MockBaseClass";
        }

        virtual int mock_method()
        {
        	return 0;
        }

        friend bool operator==(const MockBaseClass& first, const MockBaseClass& second);
    };

    bool operator==(const MockBaseClass& first, const MockBaseClass& second)
    {
        return true;
    }
}

#endif // _MOCK_BASE_CLASS_H_
