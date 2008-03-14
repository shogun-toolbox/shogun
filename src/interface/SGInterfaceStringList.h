#ifndef __INTERFACESTRINGLIST__H_
#define __INTERFACESTRINGLIST__H_

#include <typeinfo>
#include "lib/common.h"
#include "base/SGObject.h"
#include "features/StringFeatures.h"

#include "interface/SGInterfaceDataType.h"

class CSGInterfaceStringList
{
	public:
		CSGInterfaceStringList(SGInterfaceDataType idt)
		{
			m_type=idt;
		}

		CSGInterfaceStringList(T_STRING<WORD>* data, UINT len=0)
		{
			set(data, len);
		}

		CSGInterfaceStringList(T_STRING<CHAR>* data, UINT len=0)
		{
			set(data, len);
		}

		virtual inline void get(const T_STRING<WORD>*& str, UINT& len)
		{
			str=m_WORD;
			len=m_len;
		}

		virtual inline void get(const T_STRING<CHAR>*& str, UINT& len)
		{
			str=m_CHAR;
			len=m_len;
		}

		virtual inline void set(const T_STRING<WORD>* str, UINT len)
		{
			init(len);
			set_type_from_name(typeid(*str).name());
			m_WORD=str;
		}

		virtual inline void set(const T_STRING<CHAR>* str, UINT len)
		{
			init(len);
			set_type_from_name(typeid(*str).name());
			m_CHAR=str;
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
		const T_STRING<WORD>* m_WORD;
		const T_STRING<CHAR>* m_CHAR;
		UINT m_len;
		SGInterfaceDataType m_type;

	private:
		inline void init(UINT len)
		{
			m_WORD=NULL;
			m_CHAR=NULL;
			m_type=SGIDT_UNKNOWN;
			m_len=len;
		}

		inline void set_type_from_name(const CHAR* name)
		{
			if (strncmp(name, "8T_STRINGIcE", 12)==0)
				m_type=SGIDT_CHAR;
			else if (strncmp(name, "8T_STRINGItE", 12)==0)
				m_type=SGIDT_WORD;
			else
			{
				SG_WARNING("unknown type name: %s\n", name);
				m_type=SGIDT_UNKNOWN;
			}
		}
};


#endif // __INTERFACESTRINGLIST__H_
