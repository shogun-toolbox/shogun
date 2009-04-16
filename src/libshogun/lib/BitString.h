/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max Planck Society
 */
#ifndef __BITSTRING_H__
#define __BITSTRING_H__

#include <shogun/features/Alphabet.h>
#include <shogun/lib/common.h>
#include <shogun/lib/io.h>
#include <shogun/lib/MemoryMappedFile.h>
#include <shogun/lib/Mathematics.h>

/** @brief a string class embedding a string in a compact bit representation
 *
 * especially useful to compactly represent genomic DNA
 *
 * (or any other string of small alphabet size)
 *
 */
class CBitString : public CSGObject
{
	public:
		/** default constructor
		 *
		 * creates an empty Bitstring
		 *
		 * @param alpha Alphabet
		 * @param width return this many bits upon str[idx] access operations
		 */
		CBitString(EAlphabet alpha, int32_t width=1) : string(NULL), length(0), word_len(width)
		{
			alphabet=new CAlphabet(alpha);
		}

		/** destructor */
		virtual ~CBitString()
		{
			cleanup();
			SG_UNREF(alphabet);
		}

		/** free up memory */
		void cleanup()
		{
			delete[] string;
			string=NULL;
			length=0;
		}

		/** convert string of length len into bit sequence
		 *
		 * @param str string
		 * @param len length of string in bits
		 */
		void obtain_from_char(char* str, uint64_t len)
		{
			cleanup();
			uint64_t stream_len=len/sizeof(uint64_t)+1;
			string=new uint64_t[stream_len];
			length=len;

			uint64_t w=0;
			int32_t nbits=alphabet->get_num_bits();
			uint64_t j=0;
			int32_t nfit=8*sizeof(w)/nbits;
			for (uint64_t i=0; i<len; i++)
			{
				w= (w << nbits) | alphabet->remap_to_bin((uint8_t) str[j]);

				if (i % nfit == nfit-1)
				{
					string[j]=w;
					j++;
				}
			}

			if (j<stream_len)
			{
				string[j]=w;
				j++;
			}

			ASSERT(j==stream_len);
		}

		/** load fasta file as bit string
		 *
		 * @param fname filename to load from
		 * @param ignore_invalid if set to true, characters other than A,C,G,T are converted to A
		 * @return if loading was successful
		 */
		void load_fasta_file(const char* fname, bool ignore_invalid=false)
		{
			uint64_t len=0;
			uint64_t offs=0;

			CMemoryMappedFile<char> f(fname);

			uint64_t id_len=0;
			char* id=f.get_line(id_len, offs);

			if (!id_len || id[0]!='>')
				SG_ERROR("No fasta hunks (lines starting with '>') found\n");

			char* fasta=f.get_line(len, offs);
			char* s=fasta;
			uint64_t fasta_len=0;
			int32_t spanned_lines=0;

			while (true)
			{
				if (!s || len==0)
					SG_ERROR("Error reading fasta entry in line %d len=%ld", spanned_lines+1, len);

				if (s[0]=='>')
					SG_ERROR("Multiple fasta hunks (lines starting with '>') are not supported!\n");

				if (offs==f.get_size())
				{
					uint64_t w=0;
					int32_t nbits=alphabet->get_num_bits();
					uint64_t nfit=8*sizeof(w)/nbits;

					len = fasta_len-spanned_lines;
					uint64_t stream_len=len/(nfit)+1;
					string=new uint64_t[stream_len];
					length=len;

					int32_t idx=0;

					for (int32_t j=0; j<fasta_len; j++)
					{
						if (fasta[j]=='\n')
							continue;

						w= (w << nbits) | alphabet->remap_to_bin((uint8_t) fasta[j]);

						if (j % nfit == nfit-1)
						{
							string[idx]=w;
							idx++;
						}
					}

					if (idx<stream_len)
						string[idx]=w;
					break;
				}

				spanned_lines++;
				fasta_len+=len+1; // including '\n'
				s=f.get_line(len, offs);
			}

			//for (uint64_t i=0; i<length/(8*sizeof(uint64_t)); i++)
			//	CMath::display_bits(string[i], (8*sizeof(uint64_t)));


			//for (uint64_t i=0; i<2; i++)
			//	CMath::display_bits(string[i], (8*sizeof(uint64_t)));

			SG_PRINT("\n\n");
			for (uint64_t i=0; i<2; i++)
			{
				uint64_t mask=0;
				uint64_t word=string[i];
				int32_t nbits=alphabet->get_num_bits();

				for (int32_t j=0; j<nbits; j++)
					mask=(mask<<1) | (uint64_t) 1;

				for (int32_t j=0; j<sizeof(uint64_t)*8; j++)
				{
					uint64_t w=(word & mask);
					SG_PRINT("%c", alphabet->remap_to_char((uint8_t) w));
					word>>=nbits;
				}
			}
			SG_PRINT("\n\n");
		}

		/** set string of length len embedded in a uint64_t sequence
		 *
		 * @param str string
		 * @param len length of string in bits
		 */
		void set_string(uint64_t* str, uint64_t len)
		{
			cleanup();
			string=str;
			length=len;
		}

		inline uint64_t operator[](uint64_t index) const
		{
			uint64_t ws=get_sizeof_word();
			uint64_t i=index/ws;
			uint64_t j=index % ws;
			//if (
			//uint64_t x=string[i];
			return array[index];
		}


		inline uint64_t bitword_to_word_index(uint64_t bit_idx)
		{
			return bit_idx/(get_sizeof_word()*alphabet->get_num_bits());
		}

		/** return size of word in bits */
		inline uint64_t get_sizeof_word()
		{
			return 8*sizeof(uint64_t);
		}


		/** @return object name */
		inline virtual const char* get_name() const { return "BitString"; }

	private:
		/** alphabet the bit string is based on */
		CAlphabet* alphabet;
		/** the bit string */
		uint64_t* string;
		/** the length of the bit string */
		uint64_t length;
		/** the length of a word in bits */
		int32_t word_len;
};
#endif //__BITSTRING_H__
