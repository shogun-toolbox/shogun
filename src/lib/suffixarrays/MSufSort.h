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

#ifndef MSUFSORT_H
#define MSUFSORT_H

//==================================================================//
//																	//
// MSufSort	Version 2.2												//
//																	//
// Author: Michael A Maniscalco										//
// Date: Nov. 3, 2005												//
//																	//
// Notes:															//
//																	//
//==================================================================//


#include "stdio.h"
#include "Stack.h"
#include "IntroSort.h"
#include "InductionSort.h"


//==================================================================//
// Test app defines:												//
//==================================================================//

//#define SHOW_PROGRESS						// display progress during sort
//#define CHECK_SORT							// verify that sorting is correct.
//#define SORT_16_BIT_SYMBOLS					// enable 16 bit symbols.

#define USE_INDUCTION_SORTING				// enable induction sorting feature.
#define USE_ENHANCED_INDUCTION_SORTING		// enable enhanced induction sorting feature.
#define USE_TANDEM_REPEAT_SORTING			// enable the tandem repeat sorting feature.
//#define USE_ALT_SORT_ORDER					// enable alternative sorting order


#define ENDIAN_SWAP_16(value)				((value >> 8) | (value << 8))

#define SUFFIX_SORTED						0x80000000	// flag marks suffix as sorted.
#define END_OF_CHAIN						0x3ffffffe	// marks the end of a chain
#define SORTED_BY_ENHANCED_INDUCTION		0x3fffffff	// marks suffix which will be sorted by enhanced induction sort.



#ifdef SORT_16_BIT_SYMBOLS
	#define SYMBOL_TYPE		unsigned short
#else
	#define SYMBOL_TYPE		unsigned char
#endif

class MSufSort
{
public:
	MSufSort();

	virtual ~MSufSort();

	unsigned int Sort(SYMBOL_TYPE * source, unsigned int sourceLength);

	unsigned int GetElapsedSortTime();

	unsigned int GetMemoryUsage();

	unsigned int ISA(unsigned int index);

	bool VerifySort();

	static void ReverseAltSortOrder(SYMBOL_TYPE * data, unsigned int nBytes);


private:
	int CompareStrings(SYMBOL_TYPE * stringA, SYMBOL_TYPE * stringB, int len);

	bool IsTandemRepeat2();

	bool IsTandemRepeat();
	
	void PassTandemRepeat();
	
	bool IsSortedByInduction();

	bool IsSortedByEnhancedInduction(unsigned int suffixIndex);

	void ProcessSuffixesSortedByInduction();

	// MarkSuffixAsSorted
	// Sets the final inverse suffix array value for a given suffix.
	// Also invokes the OnSortedSuffix member function.
	void MarkSuffixAsSorted(unsigned int suffixIndex, unsigned int & sortedIndex);
	void MarkSuffixAsSorted2(unsigned int suffixIndex, unsigned int & sortedIndex);

	void MarkSuffixAsSortedByEnhancedInductionSort(unsigned int suffixIndex);

	// PushNewChainsOntoStack:
	// Moves all new suffix chains onto the stack of partially sorted
	// suffixes.  (makes them ready for further sub sorting).
	void PushNewChainsOntoStack(bool originalChains = false);

	void PushTandemBypassesOntoStack();

	// OnSortedSuffix:
	// Event which is invoked with each sorted suffix at the time of
	// its sorting.
	virtual void OnSortedSuffix(unsigned int suffixIndex);

	// Initialize:
	// Initializes this object just before sorting begins.
	void Initialize();

	// InitialSort:
	// This is the first sorting pass which makes the initial suffix
	// chains from the given source string.  Pushes these chains onto
	// the stack for further sorting.
	void InitialSort();

	// Value16:
	// Returns the two 8 bit symbols located
	// at positions N and N + 1 where N = the sourceIndex.
	unsigned short Value16(unsigned int sourceIndex);

	// ProcessChain:
	// Sorts the suffixes of a given chain by the next two symbols of
	// each suffix in the chain.  This creates zero or more new suffix
	// chains with each sorted by two more symbols than the original
	// chain.  Then pushes these new chains onto the chain stack for
	// further sorting.
	void ProcessNextChain();

	void AddToSuffixChain(unsigned int suffixIndex, unsigned short suffixChain);

	void AddToSuffixChain(unsigned int firstSuffixIndex, unsigned int lastSuffixIndex, unsigned short suffixChain);

	void ProcessSuffixesSortedByEnhancedInduction(unsigned short suffixId);

	void ResolveTandemRepeatsNotSortedWithInduction();

	unsigned int				m_sortTime;

	Stack<unsigned int>			m_chainMatchLengthStack;

	Stack<int>				m_chainCountStack;

	Stack<unsigned int>			m_chainHeadStack;

	unsigned int				m_endOfSuffixChain[0x10000];

	unsigned int				m_startOfSuffixChain[0x10000];

	// m_source:
	// Address of the string to sort.
	SYMBOL_TYPE *				m_source;

	// m_sourceLength:
	// The length of the string pointed to by m_source.
	unsigned int				m_sourceLength;

	unsigned int				m_sourceLengthMinusOne;

	// m_ISA:
	// The address of the working space which, when the sort is
	// completed, will contain the inverse suffix array for the 
	// source string.
	unsigned int *				m_ISA;

	// m_nextSortedSuffixValue:
	unsigned int				m_nextSortedSuffixValue;

	//
	unsigned int				m_numSortedSuffixes;

	// m_newChainIds
	// Array containing the valid chain numbers in m_newChain array.
	unsigned short				m_newChainIds[0x10000];

	// m_numNewChains:
	// The number of new suffix chain ids stored in m_numChainIds.
	unsigned int				m_numNewChains;

	Stack<InductionSortObject>	m_suffixesSortedByInduction;

	unsigned int				m_suffixMatchLength;

	unsigned int				m_currentSuffixIndex;

	// m_firstSortedPosition:
	// For use with enhanced induction sorting.  
	unsigned int				m_firstSortedPosition[0x10000];

	unsigned int				m_firstSuffixByEnhancedInductionSort[0x10000];

	unsigned int				m_lastSuffixByEnhancedInductionSort[0x10000];

	unsigned int				m_currentSuffixChainId;

	#ifdef SHOW_PROGRESS
		// ShowProgress:
		// Update the progress indicator.
		void ShowProgress();

		// m_nextProgressUpdate:
		// Indicates when to update the progress indicator.
		unsigned int			m_nextProgressUpdate;

		// m_progressUpdateIncrement:
		// Indicates how many suffixes should be sorted before
		// incrementing the progress indicator.
		unsigned int			m_progressUpdateIncrement;
	#endif


	// members used if alternate sorting order should be applied.
	SYMBOL_TYPE		m_forwardAltSortOrder[256];

	static SYMBOL_TYPE		m_reverseAltSortOrder[256];

	// for tandem repeat sorting
	bool				m_hasTandemRepeatSortedByInduction;

	unsigned int		m_firstUnsortedTandemRepeat;

	unsigned int		m_lastUnsortedTandemRepeat;

	bool				m_hasEvenLengthTandemRepeats;

	unsigned int		m_tandemRepeatDepth;

	unsigned int		m_firstSortedTandemRepeat;

	unsigned int		m_lastSortedTandemRepeat;

	unsigned int		m_tandemRepeatLength;
};





inline unsigned short MSufSort::Value16(unsigned int sourceIndex)
{
	return (sourceIndex < m_sourceLengthMinusOne) ? *(unsigned short *)(m_source + sourceIndex) : m_source[sourceIndex];
}






inline bool MSufSort::IsSortedByInduction()
{
	unsigned int n = m_currentSuffixIndex + m_suffixMatchLength - 1;

	#ifndef USE_INDUCTION_SORTING
		if (n < m_sourceLengthMinusOne)
			return false;
	#endif

	if ((m_ISA[n] & SUFFIX_SORTED) && ((m_ISA[n] & 0x3fffffff) < m_nextSortedSuffixValue))
	{
		InductionSortObject i(0, m_ISA[n], m_currentSuffixIndex);
		m_suffixesSortedByInduction.Push(i);
	}
	else
		if ((m_ISA[n + 1] & SUFFIX_SORTED) && ((m_ISA[n + 1] & 0x3fffffff) < m_nextSortedSuffixValue))
		{
			InductionSortObject i(1, m_ISA[n + 1], m_currentSuffixIndex);
			m_suffixesSortedByInduction.Push(i);
		}
		else
			return false;

	return true;
}








inline bool MSufSort::IsSortedByEnhancedInduction(unsigned int suffixIndex)
{
	if (suffixIndex > 0)
		if (m_ISA[suffixIndex - 1] == SORTED_BY_ENHANCED_INDUCTION)
			return true;
	return false;
}






inline bool MSufSort::IsTandemRepeat()
{
	#ifndef USE_TANDEM_REPEAT_SORTING
		return false;
	#else
		if ((!m_tandemRepeatDepth) && (m_currentSuffixIndex + m_suffixMatchLength) == (m_ISA[m_currentSuffixIndex] + 1))
			return true;

		#ifndef SORT_16_BIT_SYMBOLS
			if ((!m_tandemRepeatDepth) && ((m_currentSuffixIndex + m_suffixMatchLength) == (m_ISA[m_currentSuffixIndex])))
			{
				m_hasEvenLengthTandemRepeats = true;
				return false;
			}
		#endif

		return false;
	#endif
}







inline void MSufSort::PassTandemRepeat()
{
	unsigned int nextIndex;
	unsigned int lastIndex;
	// unsigned int firstIndex = m_currentSuffixIndex;

	while ((m_currentSuffixIndex + m_suffixMatchLength) == ((nextIndex = m_ISA[m_currentSuffixIndex]) + 1))
	{
		lastIndex = m_currentSuffixIndex;
		m_currentSuffixIndex = nextIndex;
	}

	if (IsSortedByInduction())
	{
		m_hasTandemRepeatSortedByInduction = true;
		m_currentSuffixIndex = m_ISA[m_currentSuffixIndex];
	}
	else
	{
		if (m_firstUnsortedTandemRepeat == END_OF_CHAIN)
			m_firstUnsortedTandemRepeat = m_lastUnsortedTandemRepeat = lastIndex;
		else
			m_lastUnsortedTandemRepeat = (m_ISA[m_lastUnsortedTandemRepeat] = lastIndex);
	}
}






inline void MSufSort::PushNewChainsOntoStack(bool originalChains)
{
	// Moves all new suffix chains onto the stack of partially sorted
	// suffixes.  (makes them ready for further sub sorting).
	#ifdef SORT_16_BIT_SYMBOLS
		unsigned int newSuffixMatchLength = m_suffixMatchLength + 1;
	#else
		unsigned int newSuffixMatchLength = m_suffixMatchLength + 2;
	#endif

	if (m_numNewChains)
	{
		if (m_hasEvenLengthTandemRepeats)
		{
			m_chainCountStack.Push(m_numNewChains - 1);
			m_chainMatchLengthStack.Push(newSuffixMatchLength);
			m_chainCountStack.Push(1);
			m_chainMatchLengthStack.Push(newSuffixMatchLength - 1);
		}
		else
		{
			m_chainCountStack.Push(m_numNewChains);
			m_chainMatchLengthStack.Push(newSuffixMatchLength);
		}

		if (m_numNewChains > 1)
			IntroSort(m_newChainIds, m_numNewChains);

		while (m_numNewChains)
		{
			unsigned short chainId = m_newChainIds[--m_numNewChains];
			chainId = ENDIAN_SWAP_16(chainId);
			// unsigned int n = m_startOfSuffixChain[chainId];
			m_chainHeadStack.Push(m_startOfSuffixChain[chainId]);
			m_startOfSuffixChain[chainId] = END_OF_CHAIN;
			m_ISA[m_endOfSuffixChain[chainId]] = END_OF_CHAIN;
		}
	}
	m_hasEvenLengthTandemRepeats = false;

	if (m_firstUnsortedTandemRepeat != END_OF_CHAIN)
	{
		// Tandem repeats with a terminating suffix that did not get
		// sorted via induction has occurred (at least once).
		// We have a suffix chain (indicated by m_firstTandemRepeatWithoutSuffix)
		// of the suffix in each tandem repeat which immediately proceeded the
		// terminating suffix in each chain.  We want to sort them relative to
		// each other and then process the tandem repeats.
		unsigned int tandemRepeatLength = m_suffixMatchLength - 1;
		unsigned int numChains = m_chainHeadStack.Count();
		m_chainHeadStack.Push(m_firstUnsortedTandemRepeat);
		m_chainCountStack.Push(1);
		m_chainMatchLengthStack.Push((m_suffixMatchLength << 1) - 1);
		m_ISA[m_lastUnsortedTandemRepeat] = END_OF_CHAIN;
		m_firstUnsortedTandemRepeat = END_OF_CHAIN;

		m_tandemRepeatDepth = 1;
		while (m_chainHeadStack.Count() > numChains)
			ProcessNextChain();

		m_suffixMatchLength = tandemRepeatLength + 1;
		ResolveTandemRepeatsNotSortedWithInduction();
		m_tandemRepeatDepth = 0;
	}

}






inline void MSufSort::AddToSuffixChain(unsigned int suffixIndex, unsigned short suffixChain)
{
	if (m_startOfSuffixChain[suffixChain] == END_OF_CHAIN)
	{
		m_endOfSuffixChain[suffixChain] = m_startOfSuffixChain[suffixChain] = suffixIndex;
		m_newChainIds[m_numNewChains++] = ENDIAN_SWAP_16(suffixChain);
	}
	else
		m_endOfSuffixChain[suffixChain] = m_ISA[m_endOfSuffixChain[suffixChain]] = suffixIndex;
}






inline void MSufSort::AddToSuffixChain(unsigned int firstSuffixIndex, unsigned int lastSuffixIndex, unsigned short suffixChain)
{
	if (m_startOfSuffixChain[suffixChain] == END_OF_CHAIN)
	{
		m_startOfSuffixChain[suffixChain] = firstSuffixIndex;
		m_endOfSuffixChain[suffixChain] = lastSuffixIndex;
		m_newChainIds[m_numNewChains++] = ENDIAN_SWAP_16(suffixChain);
	}
	else
	{
		m_ISA[m_endOfSuffixChain[suffixChain]] = firstSuffixIndex;
		m_endOfSuffixChain[suffixChain] = lastSuffixIndex;
	}
}







inline void MSufSort::OnSortedSuffix(unsigned int suffixIndex)
{
	// Event which is invoked with each sorted suffix at the time of
	// its sorting.
	m_numSortedSuffixes++;
	#ifdef SHOW_PROGRESS
		if (m_numSortedSuffixes >= m_nextProgressUpdate)
		{
			m_nextProgressUpdate += m_progressUpdateIncrement;
			ShowProgress();
		}
	#endif
}



#ifdef SORT_16_BIT_SYMBOLS

inline void MSufSort::MarkSuffixAsSorted(unsigned int suffixIndex, unsigned int  & sortedIndex)
{
	// Sets the final inverse suffix array value for a given suffix.
	// Also invokes the OnSortedSuffix member function.

	if (m_tandemRepeatDepth)
	{
		// we are processing a list of suffixes which we the second to last in tandem repeats
		// that were not terminated via induction.  These suffixes are not actually to be
		// marked as sorted yet.  Instead, they are to be linked together in sorted order.
		if (m_firstSortedTandemRepeat == END_OF_CHAIN)
			m_firstSortedTandemRepeat = m_lastSortedTandemRepeat = suffixIndex;
		else
			m_lastSortedTandemRepeat = (m_ISA[m_lastSortedTandemRepeat] = suffixIndex);
		return;
	}

	m_ISA[suffixIndex] = (sortedIndex++ | SUFFIX_SORTED);
	#ifdef SHOW_PROGRESS
		OnSortedSuffix(suffixIndex);
	#endif

	#ifdef USE_ENHANCED_INDUCTION_SORTING
		if ((suffixIndex) && (m_ISA[suffixIndex - 1] == SORTED_BY_ENHANCED_INDUCTION))
		{
			suffixIndex--;
			unsigned short symbol = Value16(suffixIndex);

			m_ISA[suffixIndex] = (m_firstSortedPosition[symbol]++ | SUFFIX_SORTED);
			#ifdef SHOW_PROGRESS
				OnSortedSuffix(suffixIndex);
			#endif

			if ((suffixIndex) && (m_ISA[suffixIndex - 1] == SORTED_BY_ENHANCED_INDUCTION))
			{
				suffixIndex--;
				symbol = ENDIAN_SWAP_16(symbol);
				if (m_firstSuffixByEnhancedInductionSort[symbol] == END_OF_CHAIN)
					m_firstSuffixByEnhancedInductionSort[symbol] = m_lastSuffixByEnhancedInductionSort[symbol] = suffixIndex;
				else
				{
					m_ISA[m_lastSuffixByEnhancedInductionSort[symbol]] = suffixIndex;
					m_lastSuffixByEnhancedInductionSort[symbol] = suffixIndex;
				}
			}
		}
	#endif
}




inline void MSufSort::MarkSuffixAsSorted2(unsigned int suffixIndex, unsigned int  & sortedIndex)
{
	// Sets the final inverse suffix array value for a given suffix.
	// Also invokes the OnSortedSuffix member function.

	if (m_tandemRepeatDepth)
	{
		// we are processing a list of suffixes which we the second to last in tandem repeats
		// that were not terminated via induction.  These suffixes are not actually to be
		// marked as sorted yet.  Instead, they are to be linked together in sorted order.
		if (m_firstSortedTandemRepeat == END_OF_CHAIN)
			m_firstSortedTandemRepeat = m_lastSortedTandemRepeat = suffixIndex;
		else
			m_lastSortedTandemRepeat = (m_ISA[m_lastSortedTandemRepeat] = suffixIndex);
		return;
	}

	m_ISA[suffixIndex] = (sortedIndex++ | SUFFIX_SORTED);
	#ifdef SHOW_PROGRESS
		OnSortedSuffix(suffixIndex);
	#endif

	#ifdef USE_ENHANCED_INDUCTION_SORTING
	if ((suffixIndex) && (m_ISA[suffixIndex - 1] == SORTED_BY_ENHANCED_INDUCTION))
	{
		unsigned short symbol = Value16(suffixIndex);
		symbol = ENDIAN_SWAP_16(symbol);
		suffixIndex--;
		if (m_firstSuffixByEnhancedInductionSort[symbol] == END_OF_CHAIN)
			m_firstSuffixByEnhancedInductionSort[symbol] = m_lastSuffixByEnhancedInductionSort[symbol] = suffixIndex;
		else
		{
			m_ISA[m_lastSuffixByEnhancedInductionSort[symbol]] = suffixIndex;
			m_lastSuffixByEnhancedInductionSort[symbol] = suffixIndex;
		}
	}
	#endif
}


#else


inline void MSufSort::MarkSuffixAsSorted(unsigned int suffixIndex, unsigned int  & sortedIndex)
{
	// Sets the final inverse suffix array value for a given suffix.
	// Also invokes the OnSortedSuffix member function.

	if (m_tandemRepeatDepth)
	{
		// we are processing a list of suffixes which we the second to last in tandem repeats
		// that were not terminated via induction.  These suffixes are not actually to be
		// marked as sorted yet.  Instead, they are to be linked together in sorted order.
		if (m_firstSortedTandemRepeat == END_OF_CHAIN)
			m_firstSortedTandemRepeat = m_lastSortedTandemRepeat = suffixIndex;
		else
			m_lastSortedTandemRepeat = (m_ISA[m_lastSortedTandemRepeat] = suffixIndex);
		return;
	}

	m_ISA[suffixIndex] = (sortedIndex++ | SUFFIX_SORTED);
	#ifdef SHOW_PROGRESS
		OnSortedSuffix(suffixIndex);
	#endif

	#ifdef USE_ENHANCED_INDUCTION_SORTING
		if ((suffixIndex) && (m_ISA[suffixIndex - 1] == SORTED_BY_ENHANCED_INDUCTION))
		{
			suffixIndex--;
			unsigned short symbol = Value16(suffixIndex);

			m_ISA[suffixIndex] = (m_firstSortedPosition[symbol]++ | SUFFIX_SORTED);
			#ifdef SHOW_PROGRESS
				OnSortedSuffix(suffixIndex);
			#endif

		if ((suffixIndex) && (m_ISA[suffixIndex - 1] == SORTED_BY_ENHANCED_INDUCTION))
		{
			suffixIndex--;
			unsigned short symbol2 = symbol;
			symbol = Value16(suffixIndex);

			m_ISA[suffixIndex] = (m_firstSortedPosition[symbol]++ | SUFFIX_SORTED);
			#ifdef SHOW_PROGRESS
				OnSortedSuffix(suffixIndex);
			#endif

			if ((suffixIndex) && (m_ISA[suffixIndex - 1] == SORTED_BY_ENHANCED_INDUCTION))
			{
				if (m_source[suffixIndex] < m_source[suffixIndex + 1])
					symbol2 = ENDIAN_SWAP_16(symbol);
				else
					symbol2 = ENDIAN_SWAP_16(symbol2);
				suffixIndex--;
				if (m_firstSuffixByEnhancedInductionSort[symbol2] == END_OF_CHAIN)
					m_firstSuffixByEnhancedInductionSort[symbol2] = m_lastSuffixByEnhancedInductionSort[symbol2] = suffixIndex;
				else
				{
					m_ISA[m_lastSuffixByEnhancedInductionSort[symbol2]] = suffixIndex;
					m_lastSuffixByEnhancedInductionSort[symbol2] = suffixIndex;
				}
			}
		}
		}
	#endif
}




inline void MSufSort::MarkSuffixAsSorted2(unsigned int suffixIndex, unsigned int  & sortedIndex)
{
	// Sets the final inverse suffix array value for a given suffix.
	// Also invokes the OnSortedSuffix member function.

	if (m_tandemRepeatDepth)
	{
		// we are processing a list of suffixes which we the second to last in tandem repeats
		// that were not terminated via induction.  These suffixes are not actually to be
		// marked as sorted yet.  Instead, they are to be linked together in sorted order.
		if (m_firstSortedTandemRepeat == END_OF_CHAIN)
			m_firstSortedTandemRepeat = m_lastSortedTandemRepeat = suffixIndex;
		else
			m_lastSortedTandemRepeat = (m_ISA[m_lastSortedTandemRepeat] = suffixIndex);
		return;
	}

	m_ISA[suffixIndex] = (sortedIndex++ | SUFFIX_SORTED);
	#ifdef SHOW_PROGRESS
		OnSortedSuffix(suffixIndex);
	#endif

	#ifdef USE_ENHANCED_INDUCTION_SORTING
	if ((suffixIndex) && (m_ISA[suffixIndex - 1] == SORTED_BY_ENHANCED_INDUCTION))
	{
		unsigned short symbol;
		if (m_source[suffixIndex] < m_source[suffixIndex + 1])
			symbol = Value16(suffixIndex);
		else
			symbol = Value16(suffixIndex + 1);
		symbol = ENDIAN_SWAP_16(symbol);
		suffixIndex--;
		if (m_firstSuffixByEnhancedInductionSort[symbol] == END_OF_CHAIN)
			m_firstSuffixByEnhancedInductionSort[symbol] = m_lastSuffixByEnhancedInductionSort[symbol] = suffixIndex;
		else
		{
			m_ISA[m_lastSuffixByEnhancedInductionSort[symbol]] = suffixIndex;
			m_lastSuffixByEnhancedInductionSort[symbol] = suffixIndex;
		}
	}
	#endif
}

#endif


inline void MSufSort::ProcessNextChain()
{
	// Sorts the suffixes of a given chain by the next two symbols of
	// each suffix in the chain.  This creates zero or more new suffix
	// chains with each sorted by two more symbols than the original
	// chain.  Then pushes these new chains onto the chain stack for
	// further sorting.
	while (--m_chainCountStack.Top() < 0)
	{
		m_chainCountStack.Pop();
		m_chainMatchLengthStack.Pop();
	}
	m_suffixMatchLength = m_chainMatchLengthStack.Top();
	m_currentSuffixIndex = m_chainHeadStack.Pop();

	#ifdef USE_ENHANCED_INDUCTION_SORTING
	if (m_chainMatchLengthStack.Count() == 1)
	{
		// one of the original buckets from InitialSort().  This is important
		// when enhanced induction sorting is enabled.
		unsigned short chainId = Value16(m_currentSuffixIndex);
		unsigned short temp = chainId;
		chainId = ENDIAN_SWAP_16(chainId);
		while (m_currentSuffixChainId <= chainId)
			ProcessSuffixesSortedByEnhancedInduction(m_currentSuffixChainId++);
		m_nextSortedSuffixValue = m_firstSortedPosition[temp];
	}
	#endif

	if (m_ISA[m_currentSuffixIndex] == END_OF_CHAIN)
		MarkSuffixAsSorted(m_currentSuffixIndex, m_nextSortedSuffixValue);  // only one suffix in bucket so it is sorted.
	else
	{
		do
		{
			if (IsTandemRepeat())
				PassTandemRepeat();
			else
				if ((m_currentSuffixIndex != END_OF_CHAIN) && (IsSortedByInduction()))
					m_currentSuffixIndex = m_ISA[m_currentSuffixIndex];
				else
					{
						unsigned int firstSuffixIndex = m_currentSuffixIndex;
						unsigned int lastSuffixIndex = m_currentSuffixIndex;
						unsigned short targetSymbol = Value16(m_currentSuffixIndex + m_suffixMatchLength);
						unsigned int nextSuffix;

						do
						{
							nextSuffix = m_ISA[lastSuffixIndex = m_currentSuffixIndex];
							if ((m_currentSuffixIndex = nextSuffix) == END_OF_CHAIN)
								break;
							else
								if (IsTandemRepeat())
								{
									PassTandemRepeat();
									break;
								}
								else
									if (IsSortedByInduction())
									{
										m_currentSuffixIndex = m_ISA[nextSuffix];
										break;
									}
						} while (Value16(m_currentSuffixIndex + m_suffixMatchLength) == targetSymbol);

					AddToSuffixChain(firstSuffixIndex, lastSuffixIndex, targetSymbol);
				}
		} while (m_currentSuffixIndex != END_OF_CHAIN);

		ProcessSuffixesSortedByInduction();
		PushNewChainsOntoStack();
	}
}






inline void MSufSort::ProcessSuffixesSortedByInduction()
{
	unsigned int numSuffixes = m_suffixesSortedByInduction.Count();
	if (numSuffixes)
	{
		InductionSortObject * objects = m_suffixesSortedByInduction.m_stack;
		if (numSuffixes > 1)
			IntroSort(objects, numSuffixes);

		if (m_hasTandemRepeatSortedByInduction)
		{
			// During the last pass some suffixes which were sorted via induction were also 
			// determined to be the terminal suffix in a tandem repeat.  So when we mark
			// the suffixes as sorted (where were sorted via induction) we make chain together
			// the preceding suffix in the tandem repeat (if there is one).
			unsigned int firstTandemRepeatIndex = END_OF_CHAIN;
			unsigned int lastTandemRepeatIndex = END_OF_CHAIN;
			unsigned int tandemRepeatLength = m_suffixMatchLength - 1;
			m_hasTandemRepeatSortedByInduction = false;

			for (unsigned int i = 0; i < numSuffixes; i++)
			{
				unsigned int suffixIndex = (objects[i].m_sortValue[1] & 0x3fffffff);
				if ((suffixIndex >= tandemRepeatLength) && (m_ISA[suffixIndex - tandemRepeatLength] == suffixIndex))
				{
					// this suffix was a terminating suffix in a tandem repeat.
					// add the preceding suffix in the tandem repeat to the list.
					if (firstTandemRepeatIndex == END_OF_CHAIN)
						firstTandemRepeatIndex = lastTandemRepeatIndex = (suffixIndex - tandemRepeatLength);
					else
						lastTandemRepeatIndex = (m_ISA[lastTandemRepeatIndex] = (suffixIndex - tandemRepeatLength));
				}
				MarkSuffixAsSorted(suffixIndex, m_nextSortedSuffixValue);
			}

			// now process each suffix in the tandem repeat list making each as sorted.
			// build a new list for tandem repeats which preceded each in the list until there are 
			// no preceding tandem suffix for any suffix in the list.

			while (firstTandemRepeatIndex != END_OF_CHAIN)
			{
				m_ISA[lastTandemRepeatIndex] = END_OF_CHAIN;
				unsigned int suffixIndex = firstTandemRepeatIndex;
				firstTandemRepeatIndex = END_OF_CHAIN;
				while (suffixIndex != END_OF_CHAIN)
				{
					if ((suffixIndex >= tandemRepeatLength) && (m_ISA[suffixIndex - tandemRepeatLength]  == suffixIndex))
					{
						// this suffix was a terminating suffix in a tandem repeat.
						// add the preceding suffix in the tandem repeat to the list.
						if (firstTandemRepeatIndex == END_OF_CHAIN)
							firstTandemRepeatIndex = lastTandemRepeatIndex = (suffixIndex - tandemRepeatLength);
						else
							lastTandemRepeatIndex = (m_ISA[lastTandemRepeatIndex] = (suffixIndex - tandemRepeatLength));
					}
					unsigned int nextSuffix = m_ISA[suffixIndex];
					MarkSuffixAsSorted(suffixIndex, m_nextSortedSuffixValue);
					suffixIndex = nextSuffix;
				}
			}
			// finished.
		}
		else
		{
			// This is the typical branch on the condition.  There were no tandem repeats
			// encountered during the last chain that were terminated with a suffix that
			// was sorted via induction.  In this case we just mark the suffixes as sorted
			// and we are done.
			for (unsigned int i = 0; i < numSuffixes; i++)
				MarkSuffixAsSorted(objects[i].m_sortValue[1] & 0x3fffffff, m_nextSortedSuffixValue);
		}
		m_suffixesSortedByInduction.Clear();
	}
}





inline void MSufSort::ProcessSuffixesSortedByEnhancedInduction(unsigned short suffixId)
{
	//
	if (m_firstSuffixByEnhancedInductionSort[suffixId] != END_OF_CHAIN)
	{
		unsigned int currentSuffixIndex = m_firstSuffixByEnhancedInductionSort[suffixId];
		unsigned int lastSuffixIndex = m_lastSuffixByEnhancedInductionSort[suffixId];
		m_firstSuffixByEnhancedInductionSort[suffixId] = END_OF_CHAIN;
		m_lastSuffixByEnhancedInductionSort[suffixId] = END_OF_CHAIN;

		do
		{
			unsigned short symbol = Value16(currentSuffixIndex);
			unsigned int nextIndex = m_ISA[currentSuffixIndex];
			MarkSuffixAsSorted2(currentSuffixIndex, m_firstSortedPosition[symbol]);
			if (currentSuffixIndex == lastSuffixIndex)
			{
				if (m_firstSuffixByEnhancedInductionSort[suffixId] == END_OF_CHAIN)
					return;
				currentSuffixIndex = m_firstSuffixByEnhancedInductionSort[suffixId];
				lastSuffixIndex = m_lastSuffixByEnhancedInductionSort[suffixId];
				m_firstSuffixByEnhancedInductionSort[suffixId] = END_OF_CHAIN;
				m_lastSuffixByEnhancedInductionSort[suffixId] = END_OF_CHAIN;
			}
			else
				currentSuffixIndex = nextIndex;
		} while (true);
	}
}




#ifdef SHOW_PROGRESS
inline void MSufSort::ShowProgress()
{
	// Update the progress indicator.
	double p = ((double)(m_numSortedSuffixes & 0x3fffffff) / m_sourceLength) * 100;
	printf("Progress: %.2f%% %c", p, 13);
}
#endif
#endif
