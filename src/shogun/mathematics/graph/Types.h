/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNTYPES_H_
#define SHOGUNTYPES_H_

#include <iosfwd>
#include <memory>
#include <string>

#include <shogun/mathematics/graph/shogun-engine_export.h>

namespace shogun
{
	namespace graph
	{
		enum class element_type
		{
			BOOLEAN = 0,
			INT8 = 1,
			INT16 = 2,
			INT32 = 3,
			INT64 = 4,
			UINT8 = 5,
			UINT16 = 6,
			UINT32 = 7,
			UINT64 = 8,
			FLOAT32 = 9,
			FLOAT64 = 10
		};

		class SHOGUN_ENGINE_EXPORT NumberType
		{
			public:
				explicit NumberType(element_type et): m_et(et) {}
				virtual ~NumberType() = default;

				friend bool operator==(const NumberType& first, const NumberType& second);
				friend bool operator!=(const NumberType& first, const NumberType& second);

				virtual bool is_signed() const = 0;
				virtual bool is_real() const = 0;
				bool is_integral() const;

				virtual bool compatible(const NumberType& other) const = 0;
				virtual std::string to_string() const = 0;
				virtual size_t size() const = 0;
				element_type type() const { return m_et; }
				operator element_type() const { return m_et; }
			private:
				const element_type m_et;
		};

		std::ostream& operator<<(std::ostream& os, const NumberType& type);

		class SHOGUN_ENGINE_EXPORT IntegerType: public NumberType
		{
			public:
				using NumberType::NumberType;
				bool is_real() const override { return false; };
		};

		class SHOGUN_ENGINE_EXPORT FloatingPointType: public NumberType
		{
			public:
				using NumberType::NumberType;

				bool is_real() const override { return true; };

				enum Precision { SINGLE, DOUBLE };
				virtual Precision precision() const = 0;
		};

		namespace detail
		{
			template <typename DERIVED, typename BASE, element_type ELEMENT_TYPE, typename C_TYPE>
			class TypeImpl: public BASE
			{
			public:
				static constexpr element_type type_id = ELEMENT_TYPE;
				using c_type = C_TYPE;

				TypeImpl(): BASE(ELEMENT_TYPE) {}

				bool compatible(const NumberType& other) const override
				{
					if (other.type() == this->type())
					{
						return true;
					}
					else if (other.type() < this->type())
					{
						if (this->is_real())
							return true;
						if (this->is_signed() == other.is_signed())
							return true;
					}
					return false;
				}

				bool is_signed() const override { return std::is_signed_v<C_TYPE>; }

				size_t size() const override { return sizeof(C_TYPE); }

				std::string to_string() const override { return DERIVED::type_name(); }
			};

			template <typename DERIVED, element_type ELEMENT_TYPE, typename C_TYPE>
			class IntegerTypeImpl : public detail::TypeImpl<DERIVED, IntegerType, ELEMENT_TYPE, C_TYPE>
			{
			};
		}

		class SHOGUN_ENGINE_EXPORT BooleanType:
			public detail::TypeImpl<BooleanType, NumberType, element_type::BOOLEAN, bool>
		{
			public:
				static constexpr const char* type_name() { return "bool"; }

				bool is_real() const override { return false; };
		};

		class SHOGUN_ENGINE_EXPORT UInt8Type:
			public detail::IntegerTypeImpl<UInt8Type, element_type::UINT8, uint8_t>
		{
			public:
				static constexpr const char* type_name() { return "uint8"; }
		};

		class SHOGUN_ENGINE_EXPORT Int8Type:
			public detail::IntegerTypeImpl<Int8Type, element_type::INT8, int8_t>
		{
			public:
				static constexpr const char* type_name() { return "int8"; }
		};

		class SHOGUN_ENGINE_EXPORT UInt16Type:
			public detail::IntegerTypeImpl<UInt16Type, element_type::UINT16, uint16_t>
		{
			public:
				static constexpr const char* type_name() { return "uint16"; }
		};

		class SHOGUN_ENGINE_EXPORT Int16Type:
			public detail::IntegerTypeImpl<Int16Type, element_type::INT16, int16_t>
		{
			public:
				static constexpr const char* type_name() { return "int16"; }
		};

		class SHOGUN_ENGINE_EXPORT UInt32Type:
			public detail::IntegerTypeImpl<UInt32Type, element_type::UINT32, uint32_t>
		{
			public:
				static constexpr const char* type_name() { return "uint32"; }
		};

		class SHOGUN_ENGINE_EXPORT Int32Type:
			public detail::IntegerTypeImpl<Int32Type, element_type::INT32, int32_t>
		{
			public:
				static constexpr const char* type_name() { return "int32"; }
		};

		class SHOGUN_ENGINE_EXPORT Int64Type:
			public detail::IntegerTypeImpl<Int64Type, element_type::INT64, int64_t>
		{
			public:
				static constexpr const char* type_name() { return "int64"; }
		};

		class SHOGUN_ENGINE_EXPORT UInt64Type:
			public detail::IntegerTypeImpl<UInt64Type, element_type::UINT64, uint64_t> {
		public:
			static constexpr const char* type_name() { return "uint64"; }
		};

		class SHOGUN_ENGINE_EXPORT Float32Type
			: public detail::TypeImpl<Float32Type, FloatingPointType, element_type::FLOAT32, float>
		{
			public:
				Precision precision() const override;
				static constexpr const char* type_name() { return "float32"; }
		};

		class SHOGUN_ENGINE_EXPORT Float64Type
			: public detail::TypeImpl<Float64Type, FloatingPointType, element_type::FLOAT64, double>
		{
			public:
				Precision precision() const override;
				static constexpr const char* type_name() { return "float64"; }
		};

		template <typename T>
		std::shared_ptr<NumberType> from()
		{
			static_assert(true, "from<T> is not implemented for the type!");
			return nullptr;
		}
		template <>
		SHOGUN_ENGINE_EXPORT std::shared_ptr<NumberType> from<bool>();
		template <>
		SHOGUN_ENGINE_EXPORT std::shared_ptr<NumberType> from<int8_t>();
		template <>
		SHOGUN_ENGINE_EXPORT std::shared_ptr<NumberType> from<int16_t>();
		template <>
		SHOGUN_ENGINE_EXPORT std::shared_ptr<NumberType> from<int32_t>();
		template <>
		SHOGUN_ENGINE_EXPORT std::shared_ptr<NumberType> from<int64_t>();
		template <>
		SHOGUN_ENGINE_EXPORT std::shared_ptr<NumberType> from<uint8_t>();
		template <>
		SHOGUN_ENGINE_EXPORT std::shared_ptr<NumberType> from<uint16_t>();
		template <>
		SHOGUN_ENGINE_EXPORT std::shared_ptr<NumberType> from<uint32_t>();
		template <>
		SHOGUN_ENGINE_EXPORT std::shared_ptr<NumberType> from<uint64_t>();
		template <>
		SHOGUN_ENGINE_EXPORT std::shared_ptr<NumberType> from<float>();
		template <>
		SHOGUN_ENGINE_EXPORT std::shared_ptr<NumberType> from<double>();

		SHOGUN_ENGINE_EXPORT std::shared_ptr<NumberType> number_type(element_type et);
	} // namespace graph
} // namespace shogun

#endif