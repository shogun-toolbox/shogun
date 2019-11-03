#ifndef _SORTED_VECTOR_PARAMETER_H
#define _SORTED_VECTOR_PARAMETER_H

#include <shogun/base/SGObject.h>
#include <vector>

namespace shogun
{
    namespace detail
    {
        class SortedVectorParameter
        {
        public:
            using ParametersContainer = std::vector<std::pair<BaseTag, AnyParameter>>;

        private:
            struct BaseTagComparator
            {
                bool operator() (ParametersContainer::const_reference p, const BaseTag& t) const { return p.first < t; }
                bool operator() (const BaseTag& t, ParametersContainer::const_reference p) const { return t < p.first; }
            };

        public:
            void create(BaseTag&& tag, AnyParameter&& parameter)
            {
                auto [has_tag, tag_it] = has(tag);
                if (has_tag)
                {
                    error("Can not register {} twice", tag.name());
                }
                auto&& kv = std::make_pair(std::move(tag), std::move(parameter));
                auto pos = std::upper_bound(map.begin(), map.end(), kv);
                map.insert(pos, std::move(kv));
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
                auto range = std::equal_range(map.begin(), map.end(), tag, cmp);
                return range.first;
            }

            auto find(const BaseTag& tag) const
            {
                auto range = std::equal_range(map.cbegin(), map.cend(), tag, cmp);
                return range.first;
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
                auto range = std::equal_range(map.begin(), map.end(), tag, cmp);
                return std::make_tuple(range.first != range.second, range.first);
            }

            std::tuple<bool, ParametersContainer::const_iterator> has(const BaseTag& tag) const
            {
                auto range = std::equal_range(map.cbegin(), map.cend(), tag, cmp);
                return std::make_tuple(range.first != range.second, range.first);
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
            BaseTagComparator cmp;
        };
    } // namespace detail
} // namespace shogun

#endif /* _SORTED_VECTOR_PARAMETER_H */
