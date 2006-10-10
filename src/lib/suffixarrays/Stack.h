/* ***** BEGIN LICENSE BLOCK *****
 * Version: MPL 1.1
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is the MSufSort suffix sorting algorithm (Version 2.2).
 *
 * The Initial Developer of the Original Code is
 * Michael A. Maniscalco
 * Portions created by the Initial Developer are Copyright (C) 2006
 * the Initial Developer. All Rights Reserved.
 *
 * Contributor(s):
 *
 *   Michael A. Maniscalco
 *
 * ***** END LICENSE BLOCK ***** */

#ifndef MSUFSORT_STACK_H
#define MSUFSORT_STACK_H

//=============================================================================================
// A quick and dirty stack class for use with the MSufSort algorithm
//
// Author: M.A. Maniscalco
// Date: 7/30/04
// email: michael@www.michael-maniscalco.com
//
// This code is free for non commercial use only.
//
//=============================================================================================

#include "memory.h"


template <class T>
class Stack 
{
public:
	Stack(unsigned int initialSize, unsigned int maxExpandSize, bool preAllocate = false):
			m_initialSize(initialSize), m_maxExpandSize(maxExpandSize), m_preAllocate(preAllocate)
	{
		Initialize();
	}

	virtual ~Stack(){SetSize(0);}

	void Push(T value);

	T & Pop();

	T & Top();

	void SetSize(unsigned int stackSize);

	void Initialize();

	unsigned int Count();

	void Clear();

	T *				m_stack;

	T *				m_stackPtr;

	T *				m_endOfStack;

	unsigned int	m_stackSize;

	unsigned int	m_initialSize;

	unsigned int	m_maxExpandSize;

	bool			m_preAllocate;
};






template <class T> 
inline void Stack<T>::Clear()
{
	m_stackPtr = m_stack;	
}




template <class T> 
inline unsigned int Stack<T>::Count()
{
	return (unsigned int)(m_stackPtr - m_stack);
}




template <class T> 
inline void Stack<T>::Initialize()
{
	m_stack = m_endOfStack = m_stackPtr = 0;
	m_stackSize = 0;
	if (m_preAllocate)
		SetSize(m_initialSize);
}




template <class T> 
inline void Stack<T>::Push(T value)
{
	if (m_stackPtr >= m_endOfStack)
	{
		unsigned int newSize = (m_stackSize < m_maxExpandSize) ? m_stackSize + m_maxExpandSize : (m_stackSize << 1);
		SetSize(newSize);
	}
	*(m_stackPtr++) = value;
}






template <class T> 
inline T & Stack<T>::Pop()
{
	return *(--m_stackPtr);
}



template <class T> 
inline T & Stack<T>::Top()
{
	return *(m_stackPtr - 1);
}





template <class T> 
inline void Stack<T>::SetSize(unsigned int stackSize)
{
	if (m_stackSize == stackSize)
		return;

	T * newStack = 0;
	if (stackSize)
	{
		newStack = new T[stackSize];
		unsigned int bytesToCopy = (unsigned int)(m_stackPtr - m_stack) * (unsigned int)sizeof(T);
		if (bytesToCopy)
			memcpy(newStack, m_stack, bytesToCopy);

		m_stackPtr = &newStack[m_stackPtr - m_stack];
		m_endOfStack = &newStack[stackSize];
		m_stackSize = stackSize;
	}

	if (m_stack)
		delete [] m_stack;
	m_stack = newStack;
}
#endif
