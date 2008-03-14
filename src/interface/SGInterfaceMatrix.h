#ifndef __INTERFACEMATRIX__H_
#define __INTERFACEMATRIX__H_

#include <typeinfo>
#include <string>

#include "lib/common.h"
#include "base/SGObject.h"
#include "features/SparseFeatures.h"

#include "interface/SGInterfaceDataType.h"

class CSGInterfaceMatrix
{
	public:
		CSGInterfaceMatrix(SGInterfaceDataType idt)
		{
			m_type=idt;
		}

		CSGInterfaceMatrix(BYTE* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(CHAR* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(DREAL* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(INT* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(SHORT* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(SHORTREAL* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(WORD* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(TSparse<BYTE>* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(TSparse<CHAR>* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(TSparse<DREAL>* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(TSparse<INT>* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(TSparse<SHORT>* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(TSparse<SHORTREAL>* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		CSGInterfaceMatrix(TSparse<WORD>* mat, UINT m, UINT n)
		{
			set(mat, m, n);
		}

		virtual inline void get(const BYTE*& mat, UINT& m, UINT& n)
		{
			mat=m_BYTE;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const CHAR*& mat, UINT& m, UINT& n)
		{
			mat=m_CHAR;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const DREAL*& mat, UINT& m, UINT& n)
		{
			mat=m_DREAL;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const INT*& mat, UINT& m, UINT& n)
		{
			mat=m_INT;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const SHORT*& mat, UINT& m, UINT& n)
		{
			mat=m_SHORT;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const SHORTREAL*& mat, UINT& m, UINT& n)
		{
			mat=m_SHORTREAL;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const WORD*& mat, UINT& m, UINT& n)
		{
			mat=m_WORD;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const TSparse<BYTE>*& mat, UINT& m, UINT& n)
		{
			mat=m_SPARSEBYTE;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const TSparse<CHAR>*& mat, UINT& m, UINT& n)
		{
			mat=m_SPARSECHAR;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const TSparse<DREAL>*& mat, UINT& m, UINT& n)
		{
			mat=m_SPARSEDREAL;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const TSparse<INT>*& mat, UINT& m, UINT& n)
		{
			mat=m_SPARSEINT;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const TSparse<SHORT>*& mat, UINT& m, UINT& n)
		{
			mat=m_SPARSESHORT;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const TSparse<SHORTREAL>*& mat, UINT& m, UINT& n)
		{
			mat=m_SPARSESHORTREAL;
			m=m_M;
			n=m_N;
		}

		virtual inline void get(const TSparse<WORD>*& mat, UINT& m, UINT& n)
		{
			mat=m_SPARSEWORD;
			m=m_M;
			n=m_N;
		}

		virtual inline void get_element(BYTE& element, UINT idx)
		{
			ASSERT(m_BYTE && idx<m_M*m_N);
			element=m_BYTE[idx];
		}

		virtual inline void get_element(CHAR& element, UINT idx)
		{
			ASSERT(m_CHAR && idx<m_M*m_N);
			element=m_CHAR[idx];
		}

		virtual inline void get_element(DREAL& element, UINT idx)
		{
			ASSERT(m_DREAL && idx<m_M*m_N);
			element=m_DREAL[idx];
		}

		virtual inline void get_element(INT& element, UINT idx)
		{
			ASSERT(m_INT && idx<m_M*m_N);
			element=m_INT[idx];
		}

		virtual inline void get_element(SHORT& element, UINT idx)
		{
			ASSERT(m_SHORT && idx<m_M*m_N);
			element=m_SHORT[idx];
		}

		virtual inline void get_element(SHORTREAL& element, UINT idx)
		{
			ASSERT(m_SHORTREAL && idx<m_M*m_N);
			element=m_SHORTREAL[idx];
		}

		virtual inline void get_element(WORD& element, UINT idx)
		{
			ASSERT(m_WORD && idx<m_M*m_N);
			element=m_WORD[idx];
		}

		virtual inline void get_element(TSparse<BYTE>& element, UINT idx)
		{
			ASSERT(m_SPARSEBYTE && idx<m_M*m_N);
			element=m_SPARSEBYTE[idx];
		}

		virtual inline void get_element(TSparse<CHAR>& element, UINT idx)
		{
			ASSERT(m_SPARSECHAR && idx<m_M*m_N);
			element=m_SPARSECHAR[idx];
		}

		virtual inline void get_element(TSparse<DREAL>& element, UINT idx)
		{
			ASSERT(m_SPARSEDREAL && idx<m_M*m_N);
			element=m_SPARSEDREAL[idx];
		}

		virtual inline void get_element(TSparse<INT>& element, UINT idx)
		{
			ASSERT(m_SPARSEINT && idx<m_M*m_N);
			element=m_SPARSEINT[idx];
		}

		virtual inline void get_element(TSparse<SHORT>& element, UINT idx)
		{
			ASSERT(m_SPARSESHORT && idx<m_M*m_N);
			element=m_SPARSESHORT[idx];
		}

		virtual inline void get_element(TSparse<SHORTREAL>& element, UINT idx)
		{
			ASSERT(m_SPARSESHORTREAL && idx<m_M*m_N);
			element=m_SPARSESHORTREAL[idx];
		}

		virtual inline void get_element(TSparse<WORD>& element, UINT idx)
		{
			ASSERT(m_SPARSEWORD && idx<m_M*m_N);
			element=m_SPARSEWORD[idx];
		}

		virtual inline bool set(const BYTE* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_BYTE=mat;

			return true;
		}

		virtual inline bool set(const CHAR* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_CHAR=mat;

			return true;
		}

		virtual inline bool set(const DREAL* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_DREAL=mat;

			return true;
		}

		virtual inline bool set(const INT* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_INT=mat;

			return true;
		}

		virtual inline bool set(const SHORT* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_SHORT=mat;

			return true;
		}

		virtual inline bool set(const SHORTREAL* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_SHORTREAL=mat;

			return true;
		}

		virtual inline bool set(const WORD* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_WORD=mat;

			return true;
		}

		virtual inline bool set(const TSparse<BYTE>* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_SPARSEBYTE=mat;

			return true;
		}

		virtual inline bool set(const TSparse<CHAR>* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_SPARSECHAR=mat;

			return true;
		}

		virtual inline bool set(const TSparse<DREAL>* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_SPARSEDREAL=mat;

			return true;
		}

		virtual inline bool set(const TSparse<INT>* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_SPARSEINT=mat;

			return true;
		}

		virtual inline bool set(const TSparse<SHORT>* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_SPARSESHORT=mat;

			return true;
		}

		virtual inline bool set(const TSparse<SHORTREAL>* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_SPARSESHORTREAL=mat;

			return true;
		}

		virtual inline bool set(const TSparse<WORD>* mat, UINT m, UINT n)
		{
			if (!mat)
				return false;

			init(m, n);
			set_type_from_name(typeid(*mat).name());
			m_SPARSEWORD=mat;

			return true;
		}

		virtual inline SGInterfaceDataType get_type()
		{
			return m_type;
		}

		virtual inline UINT get_M()
		{
			return m_M;
		}

		virtual inline UINT get_N()
		{
			return m_N;
		}

	protected:
		const BYTE* m_BYTE;
		const CHAR* m_CHAR;
		const DREAL* m_DREAL;
		const INT* m_INT;
		const SHORT* m_SHORT;
		const SHORTREAL* m_SHORTREAL;
		const WORD* m_WORD;
		const TSparse<BYTE>* m_SPARSEBYTE;
		const TSparse<CHAR>* m_SPARSECHAR;
		const TSparse<DREAL>* m_SPARSEDREAL;
		const TSparse<INT>* m_SPARSEINT;
		const TSparse<SHORT>* m_SPARSESHORT;
		const TSparse<SHORTREAL>* m_SPARSESHORTREAL;
		const TSparse<WORD>* m_SPARSEWORD;
		UINT m_M;
		UINT m_N;
		SGInterfaceDataType m_type;

	private:
		inline void init(UINT m, UINT n)
		{
			m_BYTE=NULL;
			m_CHAR=NULL;
			m_DREAL=NULL;
			m_INT=NULL;
			m_SHORT=NULL;
			m_SHORTREAL=NULL;
			m_WORD=NULL;
			m_SPARSEBYTE=NULL;
			m_SPARSECHAR=NULL;
			m_SPARSEDREAL=NULL;
			m_SPARSEINT=NULL;
			m_SPARSESHORT=NULL;
			m_SPARSESHORTREAL=NULL;
			m_SPARSEWORD=NULL;
			m_type=SGIDT_UNKNOWN;

			m_M=m;
			m_N=n;
		}

		inline void set_type_from_name(const CHAR* name)
		{
			// compiler-dependent?
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
			else if (strncmp(name, "7TSparseIhE", 11)==0)
				m_type=SGIDT_SPARSEBYTE;
			else if (strncmp(name, "7TSparseIcE", 11)==0)
				m_type=SGIDT_SPARSECHAR;
			else if (strncmp(name, "7TSparseIdE", 11)==0)
				m_type=SGIDT_SPARSEDREAL;
			else if (strncmp(name, "7TSparseIiE", 11)==0)
				m_type=SGIDT_SPARSEINT;
			else if (strncmp(name, "7TSparseIsE", 11)==0)
				m_type=SGIDT_SPARSESHORT;
			else if (strncmp(name, "7TSparseIfE", 11)==0)
				m_type=SGIDT_SPARSESHORTREAL;
			else if (strncmp(name, "7TSparseItE", 11)==0)
				m_type=SGIDT_SPARSEWORD;
			else
			{
				SG_WARNING("unknown type name: %s\n", name);
				m_type=SGIDT_UNKNOWN;
			}
		}
};

#endif // __INTERFACEMATRIX__H_
