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

#include "MSufSort.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

//=============================================================================
// MSufSort.
//=============================================================================
SYMBOL_TYPE MSufSort::m_reverseAltSortOrder[256];

MSufSort::MSufSort():m_ISA(0), m_chainHeadStack(8192, 0x20000, true), m_suffixesSortedByInduction(120000, 1000000, true),
						m_chainMatchLengthStack(8192, 0x10000, true), m_chainCountStack(8192, 0x10000, true)
{
	// constructor.
	unsigned char array[10] = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'};
	int n = 0;
	for (; n < 10; n++)
	{
		m_forwardAltSortOrder[array[n]] = n;
		m_reverseAltSortOrder[n] = array[n];
	}

	for (int i = 0; i < 256; i++)
	{
		bool unresolved = true;
		for (int j = 0; j < 10; j++)
			if (array[j] == i)
				unresolved = false;
		if (unresolved)
		{
			m_forwardAltSortOrder[i] = n;
			m_reverseAltSortOrder[n++] = i;
		}
	}
}




MSufSort::~MSufSort()
{
	// destructor.

	// delete the inverse suffix array if allocated.
	if (m_ISA)
		delete [] m_ISA;
	m_ISA = 0;
}





void MSufSort::ReverseAltSortOrder(SYMBOL_TYPE * data, unsigned int nBytes)
{
	#ifndef SORT_16_BIT_SYMBOLS
		for (unsigned int i = 0; i < nBytes; i++)
			data[i] = m_reverseAltSortOrder[data[i]];
	#endif
}




unsigned int MSufSort::GetElapsedSortTime()
{
	return m_sortTime;
}




unsigned int MSufSort::GetMemoryUsage()
{
/*
	unsigned int ret = 5 * m_sourceLength;
	ret += (m_chainStack.m_stackSize * 4);
	ret += (m_suffixesSortedByInduction.m_stackSize * 8);
	ret += sizeof(*this);
*/
	return 0;
}





unsigned int MSufSort::Sort(SYMBOL_TYPE * source, unsigned int sourceLength)
{
	///tch:
	//printf("\nIn MSufSort::Sort()\n");

	// set the member variables to the source string and its length.
	m_source = source;
	m_sourceLength = sourceLength;
	m_sourceLengthMinusOne = sourceLength - 1;
	
	Initialize();

	unsigned int start = clock();
	InitialSort();
	while (m_chainHeadStack.Count())
		ProcessNextChain();

	while (m_currentSuffixChainId <= 0xffff)
		ProcessSuffixesSortedByEnhancedInduction(m_currentSuffixChainId++);

	unsigned int finish = clock();
	m_sortTime = finish - start;
	
	///tch:
	//printf("\nFinished MSufSort::Sort()\nPress any key to continue...\n");
  //printf("%s\n",m_source);
	//system("pause");
	//getchar();
	printf("                                   %c", 13);

	return ISA(0);
}






void MSufSort::Initialize()
{
	// Initializes this object just before sorting begins.
	if (m_ISA)
		delete [] m_ISA;
	m_ISA = new unsigned int[m_sourceLength + 1];

	m_nextSortedSuffixValue = 0;
	m_numSortedSuffixes = 0;
	m_suffixMatchLength = 0;
	m_currentSuffixChainId = 0;
	m_tandemRepeatDepth = 0;
	m_firstSortedTandemRepeat = END_OF_CHAIN;
	m_hasTandemRepeatSortedByInduction = false;
	m_hasEvenLengthTandemRepeats = false;
	m_firstUnsortedTandemRepeat = END_OF_CHAIN;

	for (unsigned int i = 0; i < 0x10000; i++)
		m_startOfSuffixChain[i] = m_endOfSuffixChain[i] = m_firstSuffixByEnhancedInductionSort[i] = END_OF_CHAIN;

	for (unsigned int i = 0; i < 0x10000; i++)
		m_firstSortedPosition[i] = 0;

	m_numNewChains = 0;
	#ifdef SHOW_PROGRESS
		m_progressUpdateIncrement = (unsigned int)(m_sourceLength / 100);
		m_nextProgressUpdate = 1;
	#endif
}






void MSufSort::InitialSort()
{
	// This is the first sorting pass which makes the initial suffix
	// chains from the given source string.  Pushes these chains onto
	// the stack for further sorting.
	#ifndef SORT_16_BIT_SYMBOLS
		#ifdef USE_ALT_SORT_ORDER
			for (unsigned int suffixIndex = 0; suffixIndex < m_sourceLength; suffixIndex++)
				m_source[suffixIndex] = m_forwardAltSortOrder[m_source[suffixIndex]];
		#endif
	#endif

	#ifdef USE_ENHANCED_INDUCTION_SORTING
		m_ISA[m_sourceLength - 1] = m_ISA[m_sourceLength - 2] = SORTED_BY_ENHANCED_INDUCTION;
		m_firstSortedPosition[Value16(m_sourceLength - 1)]++;
		m_firstSortedPosition[Value16(m_sourceLength - 2)]++;
		for (int suffixIndex = m_sourceLength - 3; suffixIndex >= 0; suffixIndex--)
		{
			unsigned short symbol = Value16(suffixIndex);
				m_firstSortedPosition[symbol]++;
				#ifdef SORT_16_BIT_SYMBOLS
					unsigned short valA = ENDIAN_SWAP_16(m_source[suffixIndex]);
					unsigned short valB = ENDIAN_SWAP_16(m_source[suffixIndex + 1]);
					if ((suffixIndex == m_sourceLengthMinusOne) || (valA > valB))
						m_ISA[suffixIndex] = SORTED_BY_ENHANCED_INDUCTION;
					else
						AddToSuffixChain(suffixIndex, symbol);
				#else
					bool useEIS = false;
					if ((m_source[suffixIndex] > m_source[suffixIndex + 1]) ||
						((m_source[suffixIndex] < m_source[suffixIndex + 1]) &&
						(m_source[suffixIndex] > m_source[suffixIndex + 2])))
						useEIS = true;
					if (!useEIS)
					{
						if (m_endOfSuffixChain[symbol] == END_OF_CHAIN)
						{
							m_endOfSuffixChain[symbol] = m_startOfSuffixChain[symbol] = suffixIndex;
							m_newChainIds[m_numNewChains++] = ENDIAN_SWAP_16(symbol);
						}
						else
						{
							m_ISA[suffixIndex] = m_startOfSuffixChain[symbol];
							m_startOfSuffixChain[symbol] = suffixIndex;
						}
					}
					else
						m_ISA[suffixIndex] = SORTED_BY_ENHANCED_INDUCTION;
				#endif
		}
	#else
		for (unsigned int suffixIndex = 0; suffixIndex < m_sourceLength; suffixIndex++)
		{
			unsigned short symbol = Value16(suffixIndex);
			AddToSuffixChain(suffixIndex, symbol);
		}
	#endif


	#ifdef USE_ENHANCED_INDUCTION_SORTING
		unsigned int n = 1;
		for (unsigned int i = 0; i < 0x10000; i++)
		{
			unsigned short p = ENDIAN_SWAP_16(i);
			unsigned int temp = m_firstSortedPosition[p];
			if (temp)
			{
				m_firstSortedPosition[p] = n;
				n += temp;
			}
		}
	#endif

	MarkSuffixAsSorted(m_sourceLength, m_nextSortedSuffixValue);
	PushNewChainsOntoStack(true);
}







void MSufSort::ResolveTandemRepeatsNotSortedWithInduction()
{
	unsigned int tandemRepeatLength = m_suffixMatchLength - 1;
	unsigned int startOfFinalList = END_OF_CHAIN;

	while (m_firstSortedTandemRepeat != END_OF_CHAIN)
	{
		unsigned int stopLoopAtIndex = startOfFinalList;
		m_ISA[m_lastSortedTandemRepeat] = startOfFinalList;
		startOfFinalList = m_firstSortedTandemRepeat;

		unsigned int suffixIndex = m_firstSortedTandemRepeat;
		m_firstSortedTandemRepeat = END_OF_CHAIN;

		while (suffixIndex != stopLoopAtIndex)
		{
			if ((suffixIndex >= tandemRepeatLength) && (m_ISA[suffixIndex - tandemRepeatLength]  == suffixIndex))
			{
				if (m_firstSortedTandemRepeat == END_OF_CHAIN)
					m_firstSortedTandemRepeat = m_lastSortedTandemRepeat = (suffixIndex - tandemRepeatLength);
				else
					m_lastSortedTandemRepeat = (m_ISA[m_lastSortedTandemRepeat] = (suffixIndex - tandemRepeatLength));
			}
			suffixIndex = m_ISA[suffixIndex];
		}
	}

	m_tandemRepeatDepth--;
	if (!m_tandemRepeatDepth)
	{
		while (startOfFinalList != END_OF_CHAIN)
		{
			unsigned int next = m_ISA[startOfFinalList];
			MarkSuffixAsSorted(startOfFinalList, m_nextSortedSuffixValue);
			startOfFinalList = next;
		}
	}
	else
	{
		m_firstSortedTandemRepeat = startOfFinalList;
	}
}






unsigned int MSufSort::ISA(unsigned int index)
{
	return (m_ISA[index] & 0x3fffffff);
}





int MSufSort::CompareStrings(SYMBOL_TYPE * stringA, SYMBOL_TYPE * stringB, int len)
{
	#ifdef SORT_16_BIT_SYMBOLS
		while (len)
		{
			unsigned short valA = ENDIAN_SWAP_16(stringA[0]);
			unsigned short valB = ENDIAN_SWAP_16(stringB[0]);

			if (valA > valB)
				return 1;
			if (valA < valB)
				return -1;
			stringA++;
			stringB++;
			len--;
		}
	#else
		while (len)
		{
			if (stringA[0] > stringB[0])
				return 1;
			if (stringA[0] < stringB[0])
				return -1;
			stringA++;
			stringB++;
			len--;
		}
	#endif
	return 0;
}





bool MSufSort::VerifySort()
{
	printf("\n\nVerifying sort\n\n");
	bool error = false;
	int progressMax = m_sourceLength;
	int progressValue = 0;
	int progressUpdateStep = progressMax  / 100;
	int nextProgressUpdate = 1;

	unsigned int * suffixArray = new unsigned int[m_sourceLength];
	for (unsigned int i = 0; ((!error) && (i < m_sourceLength)); i++)
	{

		if (!(m_ISA[i] & 0x80000000))
			error = true;
		unsigned int n = (m_ISA[i] & 0x3fffffff) - 1;
		suffixArray[n] = i;
	}


	// all ok so far.
	// now compare the suffixes in lexicographically sorted order to confirm the sort was good.
	for (unsigned int suffixIndex = 0; ((!error) && (suffixIndex < (m_sourceLength - 1))); suffixIndex++)
	{
		if (++progressValue == nextProgressUpdate)
		{
			nextProgressUpdate += progressUpdateStep;
			printf("Verify sort: %.2f%% complete%c", ((double)progressValue / progressMax) * 100, 13);
		}

		SYMBOL_TYPE * ptrA = &m_source[suffixArray[suffixIndex]];
		SYMBOL_TYPE * ptrB = &m_source[suffixArray[suffixIndex + 1]];
		int maxLen = (ptrA < ptrB) ? m_sourceLength - (ptrB - m_source) : m_sourceLength - (ptrA - m_source);
		int c = CompareStrings(ptrA, ptrB, maxLen);
		if (c > 0)
			error = true;
		else
			if ((c == 0) && (ptrB > ptrA))
				error = true;
	}

	printf("                               %c", 13);
	delete [] suffixArray;
	return !error;
}
