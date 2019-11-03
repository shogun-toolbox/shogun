#ifndef _VECTOR_PARAMETER_H
#define _VECTOR_PARAMETER_H

#include <shogun/base/SGObject.h>
#include <vector>

namespace shogun
{
    namespace detail
    {
        struct VectorParameter
        {
            using ParametersContainer = std::vector<std::pair<BaseTag, AnyParameter>>;

            void create(BaseTag&& tag, AnyParameter&& parameter)
            {
                auto [has_tag, tag_it] = has(tag);
                if (has_tag)
                {
                    error("Can not register {} twice", tag.name());
                }
                map.emplace_back(std::move(tag), std::move(parameter));
            }

            void update(const BaseTag& tag, const Any& value)
            {
                auto [has_tag, tag_it] = has(tag);
                if (!has_tag)
                {
                    error(
                        "Can not update unregistered parameter {}",
                        tag.name().c_str());
                }
                tag_it->second.set_value(value);
            }

            auto find(const BaseTag& tag)
            {
                for (auto it = map.begin(); it != map.end(); it++)
                    if (it->first == tag)
                        return it;
                return map.end();
            }

            auto find(const BaseTag& tag) const
            {
                for (auto it = map.cbegin(); it != map.cend(); it++)
                    if (it->first == tag)
                        return it;
                return map.cend();
            }

            AnyParameter& at(const BaseTag& tag)
            {
                return find(tag)->second;
            }

            const AnyParameter& at(const BaseTag& tag) const
            {
                return find(tag)->second;
            }

            AnyParameter get(const BaseTag& tag) const
            {
                auto [has_tag, tag_it] = has(tag);
                if(!has_tag)
                    return AnyParameter();
                return tag_it->second;
            }

            std::tuple<bool, ParametersContainer::iterator> has(const BaseTag& tag)
            {
                auto it = find(tag);
                return std::make_tuple(it != map.end(), it);
            }

            std::tuple<bool, ParametersContainer::const_iterator> has(const BaseTag& tag) const
            {
                auto it = find(tag);
                return std::make_tuple(it != map.cend(), it);
            }

            ParametersContainer filter(ParameterProperties pprop) const
            {
                ParametersContainer result;
                std::copy_if(
                    map.cbegin(), map.cend(), std::back_inserter(result),
                    [&pprop](const auto& each) {
                        auto p = each.second.get_properties();
                        // if the filter mask is ALL, also include parameters with no set properties (NONE)
                        return p.has_property(pprop) ||
                                    (pprop==ParameterProperties::ALL &&
                                    p.compare_mask(ParameterProperties::NONE));
                    });
                return result;
            }

            ParametersContainer map;
        };
    } // namespace detail
} // namespace detail

#endif /* _VECTOR_PARAMETER_H */
