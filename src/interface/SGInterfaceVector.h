#ifndef __INTERFACEVECTOR__H_
#define __INTERFACEVECTOR__H_

#if !defined(HAVE_SWIG)

#include <typeinfo>
#include "lib/common.h"
#include "base/SGObject.h"

#include "interface/SGInterfaceDataType.h"

/*
template <class T> class CSGInterfaceVector
{
	public:
		CSGInterfaceVector(const T* data, UINT len=0)
		{
			m_data=data;
			m_len=len;
		}

		const T* get(INT& len)
		{
			len=m_len;
			return m_data;
		}

		bool set(const T* data, UINT len)
		{
			if (!data)
				return false;

			m_len=len;
			m_data=data;

			return true;
		}

		const CHAR* get_type()
		{
			if (m_data)
				return typeid(*m_data).name();
			else
				return "";
		}

	protected:
		T* m_data;
		UINT m_len;
};
*/

class CSGInterfaceVector
{
	public:
		CSGInterfaceVector(SGInterfaceDataType idt)
		{
			m_type=idt;
		}

		CSGInterfaceVector(BYTE* vec, UINT len=0)
		{
			set(vec, len);
		}

		CSGInterfaceVector(CHAR* vec, UINT len=0)
		{
			set(vec, len);
		}

		CSGInterfaceVector(DREAL* vec, UINT len=0)
		{
			set(vec, len);
		}

		CSGInterfaceVector(INT* vec, UINT len=0)
		{
			set(vec, len);
		}

		CSGInterfaceVector(SHORT* vec, UINT len=0)
		{
			set(vec, len);
		}

		CSGInterfaceVector(SHORTREAL* vec, UINT len=0)
		{
			set(vec, len);
		}

		CSGInterfaceVector(WORD* vec, UINT len=0)
		{
			set(vec, len);
		}

		virtual inline void get(BYTE*& vec, UINT& len)
		{
			vec=m_BYTE;
			len=m_len;
		}

		virtual inline void get(CHAR*& vec, UINT& len)
		{
			vec=m_CHAR;
			len=m_len;
		}

		virtual inline void get(DREAL*& vec, UINT& len)
		{
			vec=m_DREAL;
			len=m_len;
		}

		virtual inline void get(INT*& vec, INT& len)
		{
			vec=m_INT;
			len=m_len;
		}

		virtual inline void get(SHORT*& vec, UINT& len)
		{
			vec=m_SHORT;
			len=m_len;
		}

		virtual inline void get(SHORTREAL*& vec, UINT& len)
		{
			vec=m_SHORTREAL;
			len=m_len;
		}

		virtual inline void get(WORD*& vec, UINT& len)
		{
			vec=m_WORD;
			len=m_len;
		}

		virtual inline void get_element(BYTE& element, UINT idx)
		{
			ASSERT(m_BYTE && idx<m_len);
			element=m_BYTE[idx];
		}

		virtual inline void get_element(CHAR& element, UINT idx)
		{
			ASSERT(m_CHAR && idx<m_len);
			element=m_CHAR[idx];
		}

		virtual inline void get_element(DREAL& element, UINT idx)
		{
			ASSERT(m_DREAL && idx<m_len);
			element=m_DREAL[idx];
		}

		virtual inline void get_element(INT& element, UINT idx)
		{
			ASSERT(m_INT && idx<m_len);
			element=m_INT[idx];
		}

		virtual inline void get_element(SHORT& element, UINT idx)
		{
			ASSERT(m_SHORT && idx<m_len);
			element=m_SHORT[idx];
		}

		virtual inline void get_element(SHORTREAL& element, UINT idx)
		{
			ASSERT(m_SHORTREAL && idx<m_len);
			element=m_SHORTREAL[idx];
		}

		virtual inline void get_element(WORD& element, UINT idx)
		{
			ASSERT(m_WORD && idx<m_len);
			element=m_WORD[idx];
		}

		virtual inline bool set(BYTE* vec, UINT len)
		{
			if (!vec)
				return false;

			init(len);
			set_type_from_name(typeid(*vec).name());
			m_BYTE=vec;

			return true;
		}

		virtual inline bool set(CHAR* vec, UINT len)
		{
			if (!vec)
				return false;

			init(len);
			set_type_from_name(typeid(*vec).name());
			m_CHAR=vec;

			return true;
		}

		virtual inline bool set(DREAL* vec, UINT len)
		{
			if (!vec)
				return false;

			init(len);
			set_type_from_name(typeid(*vec).name());
			m_DREAL=vec;

			return true;
		}

		virtual inline bool set(INT* vec, UINT len)
		{
			if (!vec)
				return false;

			init(len);
			set_type_from_name(typeid(*vec).name());
			m_INT=vec;

			return true;
		}

		virtual inline bool set(SHORT* vec, UINT len)
		{
			if (!vec)
				return false;

			init(len);
			set_type_from_name(typeid(*vec).name());
			m_SHORT=vec;

			return true;
		}

		virtual inline bool set(SHORTREAL* vec, UINT len)
		{
			if (!vec)
				return false;

			init(len);
			set_type_from_name(typeid(*vec).name());
			m_SHORTREAL=vec;

			return true;
		}

		virtual inline bool set(WORD* vec, UINT len)
		{
			if (!vec)
				return false;

			init(len);
			set_type_from_name(typeid(*vec).name());
			m_WORD=vec;

			return true;
		}

		virtual inline SGInterfaceDataType get_type()
		{
			return m_type;
		}

		virtual inline UINT get_len()
		{
			return m_len;
		}

	protected:
		BYTE* m_BYTE;
		CHAR* m_CHAR;
		DREAL* m_DREAL;
		INT* m_INT;
		SHORT* m_SHORT;
		SHORTREAL* m_SHORTREAL;
		WORD* m_WORD;
		UINT m_len;
		SGInterfaceDataType m_type;

	private:
		inline void init(UINT len)
		{
			m_BYTE=NULL;
			m_CHAR=NULL;
			m_DREAL=NULL;
			m_INT=NULL;
			m_SHORT=NULL;
			m_SHORTREAL=NULL;
			m_WORD=NULL;
			m_type=SGIDT_UNKNOWN;

			m_len=len;
		}

		inline void set_type_from_name(const CHAR* name)
		{
			if (name[0]=='h')
				m_type=SGIDT_BYTE;
			else if (name[0]=='c')
				m_type=SGIDT_CHAR;
			else if (name[0]=='d')
				m_type=SGIDT_DREAL;
			else if (name[0]=='i')
				m_type=SGIDT_INT;
			else if (name[0]=='s')
				m_type=SGIDT_SHORT;
			else if (name[0]=='f')
				m_type=SGIDT_SHORTREAL;
			else if (name[0]=='t')
				m_type=SGIDT_WORD;
			else
			{
				SG_WARNING("unknown type name: %s\n", name);
				m_type=SGIDT_UNKNOWN;
			}
		}
};

#endif // !HAVE_SWIG
#endif // __INTERFACEVECTOR__H_
