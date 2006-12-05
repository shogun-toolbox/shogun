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

#ifndef TERNARY_INTRO_SORT_H
#define TERNARY_INTRO_SORT_H


//======================================================================//
// Class: IntroSort														//
//																		//
// Template based implementation of Introspective sorting algorithm		//
// using a ternary quicksort.											//
//																		//
// Author: M.A. Maniscalco												//
// Date: January 20, 2005												//
//																		//
//======================================================================//



// *** COMPILER WARNING DISABLED ***
// Disable a warning which appears in MSVC
// "conversion from '__w64 int' to ''"
// Just plain annoying ...  Restored at end of this file.
#ifdef WIN32
#pragma warning (disable : 4244)
#endif

#define MIN_LENGTH_FOR_QUICKSORT	32
#define MAX_DEPTH_BEFORE_HEAPSORT	128




//=====================================================================
// IntroSort class declaration 
// Notes: Any object used with this class must implement the following
// the operators:   <=, >=, ==
//=====================================================================
template <class T>
void IntroSort(T * array, unsigned int count);

template <class T>
void Partition(T * left, unsigned int count, unsigned int depth = 0);

template <class T>
T SelectPivot(T value1, T value2, T value3);

template <class T>
void Swap(T * valueA, T * valueB);

template <class T>
void InsertionSort(T * array, unsigned int count);

template <class T>
void HeapSort(T * array, int length);

template <class T>
void HeapSort(T * array, int k, int N);



template <class T>
inline void IntroSort(T * array, unsigned int count)
{
	// Public method used to invoke the sort.

	// Call quick sort partition method if there are enough
	// elements to warrant it or insertion sort otherwise.
	if (count >= MIN_LENGTH_FOR_QUICKSORT)
		Partition(array, count);
	InsertionSort(array, count);
}




template <class T> 
inline void Swap(T * valueA, T * valueB)
{
	// do the ol' "switch-a-me-do" on two values.
	T temp = *valueA; 
	*valueA = *valueB; 
	*valueB = temp;
}





template <class T> 
inline T SelectPivot(T value1, T value2, T value3)
{
	// middle of three method.
	if (value1 < value2)
		return ((value2 < value3) ? value2 : (value1 < value3) ? value3 : value1);
	return ((value1 < value3) ? value1 : (value2 < value3) ? value3 : value2); 
}





template <class T> 
inline void Partition(T * left, unsigned int count, unsigned int depth)
{
	if (++depth > MAX_DEPTH_BEFORE_HEAPSORT)
	{
		// If enough recursion has happened then we bail to heap sort since it looks
		// as if we are experiencing a 'worst case' for quick sort.  This should not
		// happen very often at all.
		HeapSort(left, count);
		return;
	}

	T * right = left + count - 1;
	T * startingLeft = left;
	T * startingRight = right;
	T * equalLeft = left;
	T * equalRight = right;

	// select the pivot value.
	T pivot = SelectPivot(left[0], right[0], left[((right - left) >> 1)]);

	// do three way partitioning.
	do
	{
		while ((left < right) && (*left <= pivot))
			if (*(left++) == pivot)
				Swap(equalLeft++, left - 1);	// equal to pivot value.  move to far left.
		
		while ((left < right) && (*right >= pivot))
			if (*(right--) == pivot)
				Swap(equalRight--, right + 1);	// equal to pivot value.  move to far right.
	
		if (left >= right)
		{
			if (left == right)
			{
				if (*left >= pivot)
					left--;
				if (*right <= pivot)
					right++;
			}
			else
			{
				left--;
				right++;
			}
			break;	// done partitioning
		}

		// left and right are ready for swaping
		Swap(left++, right--);
	} while (true);
	

	// move values that were equal to pivot from the far left into the middle.
	// these values are now placed in their final sorted position.
	if (equalLeft > startingLeft)
		while (equalLeft > startingLeft)
			Swap(--equalLeft, left--);

	// move values that were equal to pivot from the far right into the middle.
	// these values are now placed in their final sorted position.
	if (equalRight < startingRight)
		while (equalRight < startingRight)
			Swap(++equalRight, right++);

	// Calculate new partition sizes ...
	unsigned int leftSize = left - startingLeft + 1;
	unsigned int rightSize = startingRight - right + 1;

	// Partition left (less than pivot) if there are enough values to warrant it
	// otherwise do insertion sort on the values.
	if (leftSize >= MIN_LENGTH_FOR_QUICKSORT)
		Partition(startingLeft, leftSize, depth);

	// Partition right (greater than pivot) if there are enough values to warrant it
	// otherwise do insertion sort on the values.
	if (rightSize >= MIN_LENGTH_FOR_QUICKSORT)
		Partition(right, rightSize, depth);
}







template <class T> 
inline void InsertionSort(T * array, unsigned int count)
{
	// A basic insertion sort.
	if (count < 3)
	{
		if ((count == 2) && (array[0] > array[1]))
			Swap(array, array + 1);
		return;
	}

	T * ptr2, * ptr3 = array + 1, * ptr4 = array + count;

	if (array[0] > array[1])
		Swap(array, array + 1);

	while (true)
	{
		while ((++ptr3 < ptr4) && (ptr3[0] >= ptr3[-1]));

		if (ptr3 >= ptr4)
			break;

		if (ptr3[-2] <= ptr3[0])
		{ 
			if (ptr3[-1] > ptr3[0])
				Swap(ptr3, ptr3 - 1);
		}
		else
		{
			ptr2 = ptr3 - 1;
			T v = *ptr3;
			while ((ptr2 >= array) && (ptr2[0] > v))
			{
				ptr2[1] = ptr2[0];
				ptr2--;
			}
			ptr2[1] = v;
		}
	}
}





template <class T> 
inline void HeapSort(T * array, int length)
{
	// A basic heapsort.
	for (int k = length >> 1; k > 0; k--)
	    HeapSort(array, k, length);

	do
	{
		Swap(array, array + (--length));
        HeapSort(array, 1, length);
	} while (length > 1);
}





template <class T> 
inline void HeapSort(T * array, int k, int N)
{
	// A basic heapsort.
	T temp = array[k - 1];
	int n = N >> 1;

	int j = (k << 1);
	while (k <= n)
	{
        if ((j < N) && (array[j - 1] < array[j]))
	        j++;
	    if (temp >= array[j - 1])
			break;
	    else 
		{
			array[k - 1] = array[j - 1];
			k = j;
			j <<= 1;
        }
	}

    array[k - 1] = temp;
}




// Restore the default warning which appears in MSVC for
// warning #4244 which was disabled at top of this file.
#ifdef WIN32
#pragma warning (default : 4244)
#endif

#endif
