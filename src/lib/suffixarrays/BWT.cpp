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

//=============================================================================================
// A BWT class using the MSufSort suffix sorting algorithm
//
// Author: M.A. Maniscalco
// Date: 7/30/04
// email: michael@www.michael-maniscalco.com
//
// This code is free for non commercial use only.
//
//=============================================================================================




#include "BWT.h"
//#include <conio.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#define VERBOSE						
//#define CHECK_SORT



BWT::BWT()
{
	m_suffixSorter = new MSufSort;
}




BWT::~BWT()
{
	delete m_suffixSorter;
}





unsigned int BWT::Forward(SYMBOL_TYPE * data, unsigned int length)
{
	//------------------------------------------------------------------------------
	// 1.	Do the suffix sort.
	//------------------------------------------------------------------------------
	#ifdef VERBOSE
		printf("Doing MSufSort ...%c", 13);
	#endif

	int start = clock();
	unsigned int key = m_suffixSorter->Sort(data, length);
	
	#ifdef CHECK_SORT
		if (!m_suffixSorter->VerifySort())
			printf("\n*** Sort error detected ***\n");
		else
			printf("\nSort Verified\n");
	#endif

	#ifdef VERBOSE
		printf("Doing BWT ...       %c", 13);
	#endif

	//-----------------------------------------------------------------------------
	// 2.	Calculate the BWT from the suffix sort.
	//		This could also be done within the OnSortedSuffix() method during the
	//		suffix sort.  But is done after so that the MSufSort algorithm itself can
	//		be timed without the BWT portion.
	//-----------------------------------------------------------------------------

	SYMBOL_TYPE * bwtBuffer = new SYMBOL_TYPE[length];
	for (unsigned int i = 1; i <= length; i++)
	{
		unsigned int sortedPosition = m_suffixSorter->ISA(i);
		if (sortedPosition >= key)
			sortedPosition--;
		bwtBuffer[sortedPosition] = data[i - 1];
	}
	memcpy(data, bwtBuffer, length * sizeof(SYMBOL_TYPE));
	int finish = clock();

	#ifdef VERBOSE
		double elapsedTime = (double)(finish - start) / 1000;
		double msufsortTime = (double)m_suffixSorter->GetElapsedSortTime() / 1000;
		printf("    Elapsed time: %.3f seconds\n", elapsedTime);
		printf("    MSufSort time: %.3f seconds\n", msufsortTime);
		printf("    BWT time: %.3f seconds\n", elapsedTime - msufsortTime);
		printf("\n");
	#endif
	
	delete [] bwtBuffer;
	return key;
}





	
void BWT::Reverse(SYMBOL_TYPE * data, unsigned int length, unsigned int forwardKey)
{
	// reverses the blocksort transform
	#ifdef VERBOSE
		printf("Reversing the BWT ...%c", 13);
	#endif

	int start = clock();

	unsigned int * index = new unsigned int[length + 1];
	#ifdef SORT_16_BIT_SYMBOLS
		unsigned int symbolRange[0x10002];
		for (unsigned int i = 0; i < 0x10002; i++)
			symbolRange[i] = 0;
	#else
		unsigned int symbolRange[0x102];
		for (unsigned int i = 0; i < 0x102; i++)
			symbolRange[i] = 0;
	#endif
	SYMBOL_TYPE * ptr;

	//memset(symbolRange, 0, sizeof(unsigned int) * sizeof(symbolRange));
	ptr = data;

	symbolRange[0] = 1;		// our -1 symbol
	#ifdef SORT_16_BIT_SYMBOLS
		for (unsigned int i = 0; i < length; i++, ptr++)
		{
			unsigned short symbol = *ptr;
			symbol = ENDIAN_SWAP_16(symbol);
			symbolRange[symbol + 1]++;
		}
	#else
		for (unsigned int i = 0; i < length; i++, ptr++)
			symbolRange[(*ptr) + 1]++;
	#endif
	
	unsigned int n = 0;
	#ifdef SORT_16_BIT_SYMBOLS
		for (int i = 0; i < 0x10001; i++)
		{
			unsigned int temp = symbolRange[i];
			symbolRange[i] = n;
			n += temp;
		}
	#else
		for (int i = 0; i < 0x101; i++)
		{
			unsigned int temp = symbolRange[i];
			symbolRange[i] = n;
			n += temp;
		}
	#endif

	n = 0;
	index[0] = forwardKey;
	for (unsigned int i = 0; i < length; i++, n++)
	{
		if (i == forwardKey)
			n++;
		SYMBOL_TYPE symbol = data[i];
		#ifdef SORT_16_BIT_SYMBOLS
			symbol = ENDIAN_SWAP_16(symbol);
		#endif
		index[symbolRange[symbol + 1]++] = n;
	}


	n = forwardKey;
	SYMBOL_TYPE * reversedBuffer = new SYMBOL_TYPE[length];
	for (unsigned int i = 0; i < length; i++)
	{
		n = index[n];
		if (n >= forwardKey)
			reversedBuffer[i] = data[n - 1];
		else
			reversedBuffer[i] = data[n];
	}

	memcpy(data, reversedBuffer, length * sizeof(SYMBOL_TYPE));
	#ifndef SORT_16_BIT_SYMBOLS
		#ifdef USE_ALT_SORT_ORDER
			MSufSort::ReverseAltSortOrder(data, length);
		#endif
	#endif

	int finish = clock();

	#ifdef VERBOSE
		printf("Reverse BWT finished.  Elapsed time: %.2f seconds\n", (double)(finish - start) / 1000);
	#endif

	delete [] index;
	delete [] reversedBuffer;
}
