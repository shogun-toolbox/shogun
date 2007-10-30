/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Soeren Sonnenburg
 * Copyright (C) 2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CALPHABET__H__
#define _CALPHABET__H__

#include "lib/Mathematics.h"
#include "base/SGObject.h"
#include "lib/common.h"

class CAlphabet : public CSGObject
{
	public:
		CAlphabet(CHAR* alpha, INT len);
		CAlphabet(E_ALPHABET alpha);
		CAlphabet(CAlphabet* alpha);
		~CAlphabet();

		/// set alphabet and initialize mapping table (for remap)
		bool set_alphabet(E_ALPHABET alpha);

		/// get alphabet type
		inline E_ALPHABET get_alphabet()
		{
			return alphabet;
		}

		/// get number of symbols in alphabet
		inline INT get_num_symbols()
		{
			return num_symbols;
		}

		/// get number of bits necessary to store 
		/// all symbols in alphabet
		inline INT get_num_bits()
		{
			return num_bits; 
		}

		/// remap element e.g translate ACGT to 0123
		inline BYTE remap_to_bin(BYTE c)
		{
			return maptable_to_bin[c];
		}
		/// remap element e.g translate 0123 to ACGT
		inline BYTE remap_to_char(BYTE c)
		{
			return maptable_to_char[c];
		}

		/// clear histogram
		void clear_histogram();

		/// make histogram for whole string
		void add_string_to_histogram(CHAR* p, LONG len);
		void add_string_to_histogram(BYTE* p, LONG len);
		void add_string_to_histogram(SHORT* p, LONG len);
		void add_string_to_histogram(WORD* p, LONG len);
		void add_string_to_histogram(INT* p, LONG len);
		void add_string_to_histogram(UINT* p, LONG len);
		void add_string_to_histogram(LONG* p, LONG len);
		void add_string_to_histogram(ULONG* p, LONG len);

		/// add to histogram
		inline void add_byte_to_histogram(BYTE p)
		{
			histogram[(INT) p]++;
		}

		/// print histogram
		void print_histogram();

		inline void get_hist(LONG** h, INT* len)
		{
			INT hist_size=(1 << (sizeof(BYTE)*8));
			ASSERT(h && len)
			*h=(LONG*) malloc(sizeof(LONG)*hist_size);
			ASSERT(*h);
			*len=hist_size;
			ASSERT(*len);
			memcpy(*h, &histogram[0], sizeof(LONG)*hist_size);
		}

		/// get pointer to histogram
		inline const LONG* get_histogram()
        {
			//CMath::display_vector(histogram, 256, "histogram");
            return &histogram[0];
        }

		/// check whether symbols in histogram are valid in alphabet
		/// e.g. for DNA if only letters ACGT appear
		bool check_alphabet(bool print_error=true);

		/// check whether symbols in histogram ALL fit in alphabet
		bool check_alphabet_size(bool print_error=true);

		/// return number of symbols in histogram
		INT get_num_symbols_in_histogram();

		/// return maximum value in histogram
		INT get_max_value_in_histogram();

		/// return number of bits required to store all symbols in histogram
		INT get_num_bits_in_histogram();

		/// return alphabet name
		static const CHAR* get_alphabet_name(E_ALPHABET alphabet);

	protected:
		void init_map_table();
		void copy_histogram(CAlphabet* src);

	public:
		static const BYTE B_A;
		static const BYTE B_C;
		static const BYTE B_G;
		static const BYTE B_T;
		static const BYTE MAPTABLE_UNDEF;
		static const CHAR* alphabet_names[10];
	protected:
		E_ALPHABET alphabet;
		INT num_symbols;
		INT num_bits;
		BYTE valid_chars[1 << (sizeof(BYTE)*8)];
		BYTE maptable_to_bin[1 << (sizeof(BYTE)*8)];
		BYTE maptable_to_char[1 << (sizeof(BYTE)*8)];
		LONG histogram[1 << (sizeof(BYTE)*8)];
};
#endif
