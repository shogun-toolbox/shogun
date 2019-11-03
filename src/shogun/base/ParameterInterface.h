#ifndef _SGOBJECT_SELF_INTERFACE_H_
#define _SGOBJECT_SELF_INTERFACE_H_

#include <shogun/base/SGObject.h>

namespace shogun
{
    namespace detail
    {
        template <class Backend>
        class ParameterInterface
        {
            public:
                using iterator = typename Backend::ParametersContainer::iterator;
                using const_iterator = typename Backend::ParametersContainer::const_iterator;

                void create(BaseTag&& tag, AnyParameter&& parameter)
                {
                    m_backend.create(std::move(tag), std::move(parameter));
                }

                void update(const BaseTag& tag, const Any& value)
                {
                    m_backend.update(tag, value);
                }

                iterator begin() noexcept
                {
                    return m_backend.map.begin();
                }

                const_iterator begin() const noexcept
                {
                    return m_backend.map.begin();
                }

                const_iterator cbegin() const noexcept
                {
                    return m_backend.map.cbegin();
                }

                iterator end() noexcept
                {
                    return m_backend.map.end();
                }

                const_iterator end() const noexcept
                {
                    return m_backend.map.end();
                }

                const_iterator cend() const noexcept
                {
                    return m_backend.map.cend();
                }

                iterator find(const BaseTag& tag)
                {
                    return m_backend.find(tag);
                }

                const_iterator find(const BaseTag& tag) const
                {
                    return m_backend.find(tag);
                }

                AnyParameter& at(const BaseTag& tag)
                {
                    return m_backend.at(tag);
                }

                const AnyParameter& at(const BaseTag& tag) const
                {
                    return m_backend.at(tag);
                }

                AnyParameter get(const BaseTag& tag) const
                {
                    return m_backend.get(tag);
                }

                std::tuple<bool, iterator> has(const BaseTag& tag)
                {
                    return m_backend.has(tag);
                }

                std::tuple<bool, const_iterator> has(const BaseTag& tag) const
                {
                    return m_backend.has(tag);
                }

                typename Backend::ParametersContainer filter(ParameterProperties pprop) const
                {
                    return m_backend.filter(pprop);
                }

            private:
                Backend m_backend;
        };
    } // namespace detail
} // namespace shogun

#endif /* _SGOBJECT_SELF_INTERFACE_H_ */
