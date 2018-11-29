/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/lib/type_case.h>

namespace shogun {
    // define all_types global variable
    typemap all_types;

    void init_types() {
        map_type<bool>(TYPE::PT_BOOL, all_types);
        map_type<char>(TYPE::PT_CHAR, all_types);
        map_type<int8_t>(TYPE::PT_INT8, all_types);
        map_type<uint8_t>(TYPE::PT_UINT8, all_types);
        map_type<int16_t>(TYPE::PT_INT16, all_types);
        map_type<uint16_t>(TYPE::PT_UINT16, all_types);
        map_type<int32_t>(TYPE::PT_INT32, all_types);
        map_type<uint32_t>(TYPE::PT_UINT32, all_types);
        map_type<int64_t>(TYPE::PT_INT64, all_types);
        map_type<uint64_t>(TYPE::PT_UINT64, all_types);
        map_type<float32_t>(TYPE::PT_FLOAT32, all_types);
        map_type<float64_t>(TYPE::PT_FLOAT64, all_types);
        map_type<floatmax_t>(TYPE::PT_FLOATMAX, all_types);
    }

    template <typename T>
    void map_type(TYPE type, typemap& map)
    {
        map[std::type_index(typeid(T))] = type;
    }

    template <typename T>
    TYPE get_type(typemap map)
    {
        return map[std::type_index(typeid(T))];
    }

    TYPE get_type(std::type_index type, typemap map)
    {
        return map[type];
    }

    TYPE get_type(Any any, typemap map)
    {
        std::type_index t = std::type_index(any.type_info());
        return map[t];
    }

}