#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{

/** @brief Used to test the tags-parameter framework
 * Allows testing of registering new member and avoiding
 * non-registered member variables using tags framework.
 */
class CMockObject : public CSGObject
{
public:
    CMockObject() : CSGObject()
    {
        init_params();
    }

    const char* get_name() const { return "MockObject"; }

protected:
    void init_params()
    {
        float64_t decimal = 0.0;
        CKernel* kernel = NULL;
        SGVector<float64_t> *vector = NULL;
        register_param("vector", vector);
        register_param("int", m_integer);
        register_param("float", decimal);
        register_param("gaussian", kernel);
    }

private:
    int32_t m_integer = 0;
};
}