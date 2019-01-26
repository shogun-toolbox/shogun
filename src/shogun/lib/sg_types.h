#include <type_traits>
#include <shogun/lib/common.h>

// ENUMS
enum class EFeatureType
{
    F_UNKNOWN,
    F_BOOL,
    F_CHAR,
    F_BYTE,
    F_SHORT,
    F_WORD,
    F_INT,
    F_UINT,
    F_LONG,
    F_ULONG,
    F_SHORTREAL,
    F_DREAL,
    F_LONGREAL,
};

enum class EFeatureClass
{
    C_UNKNOWN = 1,
    C_DENSE = 2,
    C_SPARSE = 20,
    C_STRING = 30,
    C_COMBINED = 40,
    C_COMBINED_DOT = 60,
    C_WD = 70,
    C_SPEC = 80,
    C_WEIGHTEDSPEC = 90,
    C_POLY = 100,
    C_STREAMING_DENSE = 110,
    C_STREAMING_SPARSE = 120,
    C_STREAMING_STRING = 130,
    C_STREAMING_VW = 140,
    C_BINNED_DOT = 150,
    C_DIRECTOR_DOT = 160,
    C_LATENT = 170,
    C_MATRIX = 180,
    C_FACTOR_GRAPH = 190,
    C_INDEX = 200,
    C_SUB_SAMPLES_DENSE=300,
    C_ANY = 1000
};

enum class EFeatureProperty
{
    FP_NONE = 0,
    FP_DOT = 1,
    FP_STREAMING_DOT = 2
};

// utility structs
struct Unknown {};
struct None {};

// struct to store types
template <typename... Args>
struct Types
{
    typedef None Head;
    static constexpr int size = 0;
};

template <typename T1, typename... Args>
struct Types<T1, Args...> : Types<Args...>
{
    typedef Types<Args...> Tail;
    typedef T1 Head;
    static constexpr int size = sizeof...(Args);
};

// Type definitions
typedef	Types <long double, double, float, unsigned long, long,
        unsigned int, int, unsigned short, short, uint8_t,
        char, bool, Unknown> feature_types;

typedef Types<
        int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float32_t,
        float64_t, floatmax_t, char>
        all_primitive_types;

template<typename T1, int index, bool>
struct getTypeIndex_impl
{
};

template<typename T1, int index>
struct getTypeIndex_impl<T1, index, false>
{
    using type = None;
};

template<typename T1, int index>
struct getTypeIndex_impl<T1, index, true>
{
    using type = std::conditional_t<(T1::size == index),
            typename T1::Head,
            typename getTypeIndex_impl<typename T1::Tail, index, (T1::size > 0)>::type>;
};

template<typename T1, int index>
struct getTypeIndex
{
    using type = std::conditional_t<index >= 0,
            typename getTypeIndex_impl<T1, index, true>::type,
            None>;
};