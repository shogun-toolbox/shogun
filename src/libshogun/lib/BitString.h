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
		CBitString(EAlphabet alpha, int32_t width=1) : CSGObject(), string(NULL), length(0)
		{
			alphabet=new CAlphabet(alpha);
			int32_t nbits=alphabet->get_num_bits();
			word_len=width*nbits;
			mask=0;

			for (int32_t j=0; j<word_len; j++)
				mask=(mask<<1) | (uint64_t) 1;
			mask<<=sizeof(uint64_t)*8-word_len;
		}

		/** destructor */
		~CBitString()
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
			cleanup();

			uint64_t len=0;
			uint64_t offs=0;

			CMemoryMappedFile<char> f(fname);

			uint64_t id_len=0;
			char* id=f.get_line(id_len, offs);

			if (!id_len || id[0]!='>')
				SG_SERROR("No fasta hunks (lines starting with '>') found\n");

			if (offs==f.get_size())
				SG_SERROR("Empty file?\n");

			char* fasta=NULL;
			char* s=NULL;
			int32_t spanned_lines=0;
			int32_t fasta_len=0;

			while (true)
			{
				s=f.get_line(len, offs);

				if (!fasta)
					fasta=s;

				if (!s || len==0)
					SG_SERROR("Error reading fasta entry in line %d len=%ld", spanned_lines+1, len);

				if (s[0]=='>')
					SG_SERROR("Multiple fasta hunks (lines starting with '>') are not supported!\n");

				if (offs==f.get_size())
				{
					offs-=len+1; // seek to beginning
					fasta_len+=len;

					uint64_t w=0;
					int32_t nbits=alphabet->get_num_bits();
					uint64_t nfit=8*sizeof(w)/nbits;

					len = fasta_len-spanned_lines;
					uint64_t stream_len=len/(nfit)+1;
					string=new uint64_t[stream_len];
					length=len;

					int32_t idx=0;
					int32_t k=0;

					for (int32_t j=0; j<fasta_len; j++, k++)
					{
						if (fasta[j]=='\n')
						{
							k--;
							continue;
						}

						w= (w << nbits) | alphabet->remap_to_bin((uint8_t) fasta[j]);

						if (k % nfit == nfit-1)
						{
							string[idx]=w;
							w=0;
							idx++;
						}
					}

					if (idx<stream_len)
						string[idx]=w<<(nbits*(nfit - k%nfit));

					break;
				}

				spanned_lines++;
				fasta_len+=len+1; // including '\n'
			}
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

		inline uint64_t condense(uint64_t bits, uint64_t mask) const
			{
				uint64_t res = 0 ;
				uint64_t m=mask ;
				uint64_t b=bits ;
				char cnt=0 ;
				
				char ar[256][256] ;

				for (int i=0; i<8; i++)
				{
					//fprintf(stdout, "%i %lx %lx %lx %i\n", i, res, m, b, (int)cnt) ;
					if (m&1)
						res = res>>8 | ar[b&255][m&255] ;
					//else
					//	cnt++ ;
					m=m>>8 ;
					b=b>>8 ;
				}
				res=res>>cnt ;
				//fprintf(stdout, "%lx %lx %lx\n", res, m, b) ;
				
				//res = res & bits & mask ;
				
				return res ;
			}
		

		inline uint64_t operator[](uint64_t index) const
		{
			ASSERT(index<length);

			uint64_t bitindex=alphabet->get_num_bits()*index;
			int32_t ws=8*sizeof(uint64_t);
			uint64_t i=bitindex/ws;
			int32_t j=bitindex % ws;
			int32_t missing=word_len-(ws-j);

			//SG_SPRINT("i=%lld j=%d ws=%d word_len=%d missing=%d left=%llx shift=%d\n", i, j, ws, word_len, missing, ( string[i] << j ) & mask, ws-word_len);
			uint64_t res= ((string[i] << j) & mask ) >> (ws-word_len);

			if (missing>0)
				res|= ( string[i+1] >> (ws-missing) );

			//res = condense(res, 1<<31|1<<24|1<<10|1<<8|1<<4|1<<2|1<<1|1) ;

			return res;
		}

		inline uint64_t get_length() const { return length-word_len/alphabet->get_num_bits()+1; }

		/** @return object name */
		inline virtual const char* get_name() const { return "BitString"; }

	private:
		/** alphabet the bit string is based on */
		CAlphabet* alphabet;
		/** the bit string */
		uint64_t* string;
		/** the length of the bit string */
		uint64_t length;
		/** length of a word in bits */
		int32_t word_len;
		/** mask */
		uint64_t mask;
};
#endif //__BITSTRING_H__
