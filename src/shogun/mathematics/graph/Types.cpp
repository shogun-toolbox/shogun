#include <shogun/mathematics/graph/Types.h>

#include <ostream>

using namespace shogun::graph;

bool NumberType::is_integral() const
{
    return !is_real();
}

bool shogun::graph::operator==(const NumberType& left, const NumberType& right)
{
    if (&left == &right)
    {
        return true;
    }
    else if (left.type() == right.type())
    {
        return true;
    }
    return false;
}

bool shogun::graph::operator!=(const NumberType& left, const NumberType& right)
{
    return !(left == right);
}

std::ostream& operator<<(std::ostream& os, const NumberType& type)
{
    os << type.to_string();
    return os;
}

FloatingPointType::Precision Float32Type::precision() const
{
    return FloatingPointType::SINGLE;
}

FloatingPointType::Precision Float64Type::precision() const
{
    return FloatingPointType::DOUBLE;
}

#define TYPE_FACTORY(NAME, KLASS)                                           \
  std::shared_ptr<NumberType> NAME()                                        \
  {                                                                         \
    static std::shared_ptr<NumberType> result = std::make_shared<KLASS>();  \
    return result;                                                          \
  }

TYPE_FACTORY(boolean, BooleanType)
TYPE_FACTORY(int8, Int8Type)
TYPE_FACTORY(uint8, UInt8Type)
TYPE_FACTORY(int16, Int16Type)
TYPE_FACTORY(uint16, UInt16Type)
TYPE_FACTORY(int32, Int32Type)
TYPE_FACTORY(uint32, UInt32Type)
TYPE_FACTORY(int64, Int64Type)
TYPE_FACTORY(uint64, UInt64Type)
TYPE_FACTORY(float32, Float32Type)
TYPE_FACTORY(float64, Float64Type)

namespace shogun
{
    namespace graph
    {
        template <>
        std::shared_ptr<NumberType> from<bool>()
        {
            return boolean();
        }
        template <>
        std::shared_ptr<NumberType> from<float>()
        {
            return float32();
        }
        template <>
        std::shared_ptr<NumberType> from<double>()
        {
            return float64();
        }
        template <>
        std::shared_ptr<NumberType> from<int8_t>()
        {
            return int8();
        }
        template <>
        std::shared_ptr<NumberType> from<int16_t>()
        {
            return int16();
        }
        template <>
        std::shared_ptr<NumberType> from<int32_t>()
        {
            return int32();
        }
        template <>
        std::shared_ptr<NumberType> from<int64_t>()
        {
            return int64();
        }
        template <>
        std::shared_ptr<NumberType> from<uint8_t>()
        {
            return uint8();
        }
        template <>
        std::shared_ptr<NumberType> from<uint16_t>()
        {
            return uint16();
        }
        template <>
        std::shared_ptr<NumberType> from<uint32_t>()
        {
            return uint32();
        }
        template <>
        std::shared_ptr<NumberType> from<uint64_t>()
        {
            return uint64();
        }

        std::shared_ptr<NumberType> number_type(element_type et)
        {
            switch(et)
            {
                case element_type::BOOLEAN:
                    return boolean();
                case element_type::INT8:
                    return int8();
                case element_type::INT16:
                    return int16();
                case element_type::INT32:
                    return int32();
                case element_type::INT64:
                    return int64();
                case element_type::UINT8:
                    return uint8();
                case element_type::UINT16:
                    return uint16();
                case element_type::UINT32:
                    return uint32();
                case element_type::UINT64:
                    return uint64();
                case element_type::FLOAT32:
                    return float32();
                case element_type::FLOAT64:
                    return float64();
            }
            throw std::invalid_argument("Unknown type");
        }
    }
}
