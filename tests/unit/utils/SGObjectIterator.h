#ifndef __SG_OBJECT_ITERATOR_H__
#define __SG_OBJECT_ITERATOR_H__

#include <iterator>
#include <set>
#include <string>
#include <shogun/base/class_list.h>

namespace
{
// to have a type for non-template SGObject classes
struct untemplated_sgobject
{
};

/** Returns primitive type value for template parameter */
template <class T>
static EPrimitiveType sg_primitive_type();
template <class T>
static std::string sg_primitive_type_string();

// stringizing the result of expansion of a macro argument used below
// https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html
#define STRING__(s) #s
#define XSTRING__(s) STRING__(s)

#define SG_PRIMITIVE_TYPE(T, ptype)                                            \
	template <>                                                                \
	EPrimitiveType sg_primitive_type<T>()                                      \
	{                                                                          \
		return ptype;                                                          \
	}                                                                          \
                                                                               \
	template <>                                                                \
	std::string sg_primitive_type_string<T>()                                  \
	{                                                                          \
		return std::string(XSTRING__(ptype));                                  \
	}

SG_PRIMITIVE_TYPE(bool, PT_BOOL);
SG_PRIMITIVE_TYPE(char, PT_CHAR);
SG_PRIMITIVE_TYPE(int8_t, PT_INT8);
SG_PRIMITIVE_TYPE(uint8_t, PT_UINT8);
SG_PRIMITIVE_TYPE(int16_t, PT_INT16);
SG_PRIMITIVE_TYPE(int32_t, PT_INT32);
SG_PRIMITIVE_TYPE(int64_t, PT_INT64);
SG_PRIMITIVE_TYPE(float32_t, PT_FLOAT32);
SG_PRIMITIVE_TYPE(float64_t, PT_FLOAT64);
SG_PRIMITIVE_TYPE(floatmax_t, PT_FLOATMAX);
SG_PRIMITIVE_TYPE(untemplated_sgobject, PT_NOT_GENERIC);
#undef SG_PRIMITIVE_TYPE
#undef XSTRING__
#undef STRING__

/** Helper to write c++11 style loops in tests that cover all Shogun classes */
template <typename T>
class sg_object_iterator
{
public:
	sg_object_iterator() : sg_object_iterator(available_objects())
	{
	}

	sg_object_iterator(std::set<std::string> class_names)
	{
		m_class_names = class_names;
	}

	sg_object_iterator ignore(std::set<std::string> ignores)
	{
		std::set<std::string> diff;
		std::set_difference(
		    m_class_names.begin(), m_class_names.end(), ignores.begin(),
		    ignores.end(), std::inserter(diff, diff.begin()));
		return sg_object_iterator(diff);
	}

	class Iterator : public std::iterator<std::input_iterator_tag, T>
	{
	public:
		Iterator(
		    std::set<std::string>::iterator it,
		    std::set<std::string>::iterator end)
		    : m_it(it), m_end(end), m_obj(nullptr)
		{
			create_and_forward_if_necessary();
		}
		Iterator(const Iterator& other)
		    : m_it(other.m_it), m_end(other.m_end), m_obj(other.m_obj)
		{
			create_and_forward_if_necessary();
		}
		Iterator(Iterator&& other)
		    : m_it(other.m_it), m_end(other.m_end), m_obj(other.m_obj)
		{
			create_and_forward_if_necessary();
		}
		~Iterator()
		{
		}

		Iterator& operator=(const Iterator&) = delete;
		Iterator& operator++()
		{
			do
			{
				++m_it;

				if (m_it == m_end)
					return *this;

				m_obj = create_obj();
			} while (!m_obj);

			return *this;
		}
		std::shared_ptr<SGObject> operator*()
		{
			return m_obj;
		}
		bool operator!=(const Iterator& other) const
		{
			return this->m_it != other.m_it;
		}

	private:
		auto create_obj()
		{
			return shogun::create((*m_it).c_str(), sg_primitive_type<T>());
		}

		void create_and_forward_if_necessary()
		{
			if (m_it == m_end)
				return;

			m_obj = create_obj();
			while (!m_obj && m_it != m_end)
			{
				this->operator++();
			}
		}

		std::set<std::string>::iterator m_it;
		std::set<std::string>::iterator m_end;
		std::shared_ptr<SGObject> m_obj;
	};

	Iterator begin() const
	{
		return Iterator(m_class_names.begin(), m_class_names.end());
	}
	Iterator end() const
	{
		return Iterator(m_class_names.end(), m_class_names.end());
	}

	std::set<std::string> m_class_names;
};

} // namespace shogun
#endif //__SG_OBJECT_ITERATOR_H__
