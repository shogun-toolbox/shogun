#ifndef __SG_SOME_H__
#define __SG_SOME_H__

#ifdef HAVE_CXX11
#include <memory>

#include <shogun/base/SGObject.h>

namespace shogun
{

    /** Shogun synonym for the std::shared_ptr 
     */
    template <typename T>
    class Some : public std::shared_ptr<T>
    {
        public:
            Some(const std::shared_ptr<T>& shared)
                : std::shared_ptr<T>(shared)
            {

            }
    };

    /** Creates an instance of any class
     * that is wrapped with a shared pointer like
     * structure @ref Some
     *
     */
    template <typename T, class... Args>
    Some<T> some(Args&&... args)
    {
        T* ptr = new T(args...);
        SG_REF(ptr);
        return std::shared_ptr<T>(ptr, [](T* p) { SG_UNREF(p); });
    }

};

#endif /* HAVE_CXX11 */
#endif /* __SG_SOME_H__ */
