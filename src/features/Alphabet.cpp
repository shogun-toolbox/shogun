/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2008 Soeren Sonnenburg
 * Copyright (C) 2006-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <string.h>
#include <math.h>

#include "features/Alphabet.h"
#include "lib/io.h"

//define numbers for the bases 
const uint8_t CAlphabet::B_A=0;
const uint8_t CAlphabet::B_C=1;
const uint8_t CAlphabet::B_G=2;
const uint8_t CAlphabet::B_T=3;
const uint8_t CAlphabet::MAPTABLE_UNDEF=0xff;
const char* CAlphabet::alphabet_names[11]={"DNA", "RAWDNA", "RNA", "PROTEIN", "ALPHANUM", "CUBE", "RAW", "IUPAC_NUCLEIC_ACID", "IUPAC_AMINO_ACID", "NONE", "UNKNOWN"};

CAlphabet::CAlphabet(char* al, int32_t len)
: CSGObject()
{
	EAlphabet alpha=NONE;

	if (len>=(int32_t) strlen("DNA") && !strncmp(al, "DNA", strlen("DNA")))
		alpha = DNA;
	else if (len>=(int32_t) strlen("RAWDNA") && !strncmp(al, "RAWDNA", strlen("RAWDNA")))
		alpha = RAWDNA;
	else if (len>=(int32_t) strlen("RNA") && !strncmp(al, "RNA", strlen("RNA")))
		alpha = RNA;
	else if (len>=(int32_t) strlen("PROTEIN") && !strncmp(al, "PROTEIN", strlen("PROTEIN")))
		alpha = PROTEIN;
	else if (len>=(int32_t) strlen("ALPHANUM") && !strncmp(al, "ALPHANUM", strlen("ALPHANUM")))
		alpha = ALPHANUM;
	else if (len>=(int32_t) strlen("CUBE") && !strncmp(al, "CUBE", strlen("CUBE")))
		alpha = CUBE;
	else if ((len>=(int32_t) strlen("BYTE") && !strncmp(al, "BYTE", strlen("BYTE"))) || 
			(len>=(int32_t) strlen("RAW") && !strncmp(al, "RAW", strlen("RAW"))))
		alpha = RAWBYTE;
	else if (len>=(int32_t) strlen("IUPAC_NUCLEIC_ACID") && !strncmp(al, "IUPAC_NUCLEIC_ACID", strlen("IUPAC_NUCLEIC_ACID")))
		alpha = IUPAC_NUCLEIC_ACID;
	else if (len>=(int32_t) strlen("IUPAC_AMINO_ACID") && !strncmp(al, "IUPAC_AMINO_ACID", strlen("IUPAC_AMINO_ACID")))
		alpha = IUPAC_AMINO_ACID;
	else {
      SG_ERROR( "unknown alphabet %s\n", al);
   }
	
	set_alphabet(alpha);
}

CAlphabet::CAlphabet(EAlphabet alpha)
: CSGObject()
{
	set_alphabet(alpha);
}

CAlphabet::CAlphabet(CAlphabet* a)
: CSGObject()
{
	ASSERT(a);
	set_alphabet(a->get_alphabet());
	copy_histogram(a);
}

CAlphabet::~CAlphabet()
{
}

bool CAlphabet::set_alphabet(EAlphabet alpha)
{
	bool result=true;
	alphabet=alpha;

	switch (alphabet)
	{
		case DNA:
		case RAWDNA:
			num_symbols = 4;
			break;
		case RNA:
			num_symbols = 4;
			break;
		case PROTEIN:
			num_symbols = 26;
			break;
		case ALPHANUM:
			num_symbols = 36;
			break;
		case CUBE:
			num_symbols = 6;
			break;
		case RAWBYTE:
			num_symbols = 256;
			break;
		case IUPAC_NUCLEIC_ACID:
			num_symbols = 16;
			break;
		case IUPAC_AMINO_ACID:
			num_symbols = 23;
			break;
		case NONE:
			num_symbols = 0;
			break;
		default:
			num_symbols = 0;
			result=false;
			break;
	}

	num_bits=(int32_t) ceil(log((double) num_symbols)/log((double) 2));
	init_map_table();
    clear_histogram();

	SG_DEBUG( "initialised alphabet %s\n", get_alphabet_name(alphabet));

	return result;
}

void CAlphabet::init_map_table()
{
	int32_t i;
	for (i=0; i<(1<<(8*sizeof(uint8_t))); i++)
	{
		maptable_to_bin[i] = MAPTABLE_UNDEF;
		maptable_to_char[i] = MAPTABLE_UNDEF;
		valid_chars[i] = 0;
	}

	switch (alphabet)
	{
		case CUBE:
			valid_chars[(uint8_t) '1']=1;
			valid_chars[(uint8_t) '2']=1;
			valid_chars[(uint8_t) '3']=1;
			valid_chars[(uint8_t) '4']=1;	
			valid_chars[(uint8_t) '5']=1;	
			valid_chars[(uint8_t) '6']=1;	//Translation '123456' -> 012345

			maptable_to_bin[(uint8_t) '1']=0;
			maptable_to_bin[(uint8_t) '2']=1;
			maptable_to_bin[(uint8_t) '3']=2;
			maptable_to_bin[(uint8_t) '4']=3;	
			maptable_to_bin[(uint8_t) '5']=4;	
			maptable_to_bin[(uint8_t) '6']=5;	//Translation '123456' -> 012345

			maptable_to_char[(uint8_t) 0]='1';
			maptable_to_char[(uint8_t) 1]='2';
			maptable_to_char[(uint8_t) 2]='3';
			maptable_to_char[(uint8_t) 3]='4';
			maptable_to_char[(uint8_t) 4]='5';
			maptable_to_char[(uint8_t) 5]='6';	//Translation 012345->'123456'
			break;

		case PROTEIN:
			{
				int32_t skip=0 ;
				for (i=0; i<21; i++)
				{
					if (i==1) skip++ ;
					if (i==8) skip++ ;
					if (i==12) skip++ ;
					if (i==17) skip++ ;
					valid_chars['A'+i+skip]=1;
					maptable_to_bin['A'+i+skip]=i ;
					maptable_to_char[i]='A'+i+skip ;
				} ;                   //Translation 012345->acde...xy -- the protein code
			} ;
			break;

		case ALPHANUM:
			{
				for (i=0; i<26; i++)
				{
					valid_chars['A'+i]=1;
					maptable_to_bin['A'+i]=i ;
					maptable_to_char[i]='A'+i ;
				} ;
				for (i=0; i<10; i++)
				{
					valid_chars['0'+i]=1;
					maptable_to_bin['0'+i]=26+i ;
					maptable_to_char[26+i]='0'+i ;
				} ;        //Translation 012345->acde...xy0123456789
			} ;
			break;

		case RAWBYTE:
			{
				//identity
				for (i=0; i<256; i++)
				{
					valid_chars[i]=1;
					maptable_to_bin[i]=i;
					maptable_to_char[i]=i;
				}
			}
			break;

		case DNA:
			valid_chars[(uint8_t) 'A']=1;
			valid_chars[(uint8_t) 'C']=1;
			valid_chars[(uint8_t) 'G']=1;
			valid_chars[(uint8_t) 'T']=1;	

			maptable_to_bin[(uint8_t) 'A']=B_A;
			maptable_to_bin[(uint8_t) 'C']=B_C;
			maptable_to_bin[(uint8_t) 'G']=B_G;
			maptable_to_bin[(uint8_t) 'T']=B_T;	

			maptable_to_char[B_A]='A';
			maptable_to_char[B_C]='C';
			maptable_to_char[B_G]='G';
			maptable_to_char[B_T]='T';
			break;
		case RAWDNA:
			{
				//identity
				for (i=0; i<4; i++)
				{
					valid_chars[i]=1;
					maptable_to_bin[i]=i;
					maptable_to_char[i]=i;
				}
			}
			break;

		case RNA:
			valid_chars[(uint8_t) 'A']=1;
			valid_chars[(uint8_t) 'C']=1;
			valid_chars[(uint8_t) 'G']=1;
			valid_chars[(uint8_t) 'U']=1;	

			maptable_to_bin[(uint8_t) 'A']=B_A;
			maptable_to_bin[(uint8_t) 'C']=B_C;
			maptable_to_bin[(uint8_t) 'G']=B_G;
			maptable_to_bin[(uint8_t) 'U']=B_T;	

			maptable_to_char[B_A]='A';
			maptable_to_char[B_C]='C';
			maptable_to_char[B_G]='G';
			maptable_to_char[B_T]='U';
			break;

		case IUPAC_NUCLEIC_ACID:
			valid_chars[(uint8_t) 'A']=1; // A	Adenine
			valid_chars[(uint8_t) 'C']=1; // C	Cytosine
			valid_chars[(uint8_t) 'G']=1; // G	Guanine
			valid_chars[(uint8_t) 'T']=1; // T	Thymine
			valid_chars[(uint8_t) 'U']=1; // U	Uracil
			valid_chars[(uint8_t) 'R']=1; // R	Purine (A or G)
			valid_chars[(uint8_t) 'Y']=1; // Y	Pyrimidine (C, T, or U)
			valid_chars[(uint8_t) 'M']=1; // M	C or A
			valid_chars[(uint8_t) 'K']=1; // K	T, U, or G
			valid_chars[(uint8_t) 'W']=1; // W	T, U, or A
			valid_chars[(uint8_t) 'S']=1; // S	C or G
			valid_chars[(uint8_t) 'B']=1; // B	C, T, U, or G (not A)
			valid_chars[(uint8_t) 'D']=1; // D	A, T, U, or G (not C)
			valid_chars[(uint8_t) 'H']=1; // H	A, T, U, or C (not G)
			valid_chars[(uint8_t) 'V']=1; // V	A, C, or G (not T, not U)
			valid_chars[(uint8_t) 'N']=1; // N	Any base (A, C, G, T, or U)

			maptable_to_bin[(uint8_t) 'A']=0; // A	Adenine
			maptable_to_bin[(uint8_t) 'C']=1; // C	Cytosine
			maptable_to_bin[(uint8_t) 'G']=2; // G	Guanine
			maptable_to_bin[(uint8_t) 'T']=3; // T	Thymine
			maptable_to_bin[(uint8_t) 'U']=4; // U	Uracil
			maptable_to_bin[(uint8_t) 'R']=5; // R	Purine (A or G)
			maptable_to_bin[(uint8_t) 'Y']=6; // Y	Pyrimidine (C, T, or U)
			maptable_to_bin[(uint8_t) 'M']=7; // M	C or A
			maptable_to_bin[(uint8_t) 'K']=8; // K	T, U, or G
			maptable_to_bin[(uint8_t) 'W']=9; // W	T, U, or A
			maptable_to_bin[(uint8_t) 'S']=10; // S	C or G
			maptable_to_bin[(uint8_t) 'B']=11; // B	C, T, U, or G (not A)
			maptable_to_bin[(uint8_t) 'D']=12; // D	A, T, U, or G (not C)
			maptable_to_bin[(uint8_t) 'H']=13; // H	A, T, U, or C (not G)
			maptable_to_bin[(uint8_t) 'V']=14; // V	A, C, or G (not T, not U)
			maptable_to_bin[(uint8_t) 'N']=15; // N	Any base (A, C, G, T, or U)

			maptable_to_char[0]=(uint8_t) 'A'; // A	Adenine
			maptable_to_char[1]=(uint8_t) 'C'; // C	Cytosine
			maptable_to_char[2]=(uint8_t) 'G'; // G	Guanine
			maptable_to_char[3]=(uint8_t) 'T'; // T	Thymine
			maptable_to_char[4]=(uint8_t) 'U'; // U	Uracil
			maptable_to_char[5]=(uint8_t) 'R'; // R	Purine (A or G)
			maptable_to_char[6]=(uint8_t) 'Y'; // Y	Pyrimidine (C, T, or U)
			maptable_to_char[7]=(uint8_t) 'M'; // M	C or A
			maptable_to_char[8]=(uint8_t) 'K'; // K	T, U, or G
			maptable_to_char[9]=(uint8_t) 'W'; // W	T, U, or A
			maptable_to_char[10]=(uint8_t) 'S'; // S	C or G
			maptable_to_char[11]=(uint8_t) 'B'; // B	C, T, U, or G (not A)
			maptable_to_char[12]=(uint8_t) 'D'; // D	A, T, U, or G (not C)
			maptable_to_char[13]=(uint8_t) 'H'; // H	A, T, U, or C (not G)
			maptable_to_char[14]=(uint8_t) 'V'; // V	A, C, or G (not T, not U)
			maptable_to_char[15]=(uint8_t) 'N'; // N	Any base (A, C, G, T, or U)
			break;

		case IUPAC_AMINO_ACID:
			valid_chars[(uint8_t) 'A']=0;  //A	Ala	Alanine
			valid_chars[(uint8_t) 'R']=1;  //R	Arg	Arginine
			valid_chars[(uint8_t) 'N']=2;  //N	Asn	Asparagine
			valid_chars[(uint8_t) 'D']=3;  //D	Asp	Aspartic acid
			valid_chars[(uint8_t) 'C']=4;  //C	Cys	Cysteine
			valid_chars[(uint8_t) 'Q']=5;  //Q	Gln	Glutamine
			valid_chars[(uint8_t) 'E']=6;  //E	Glu	Glutamic acid
			valid_chars[(uint8_t) 'G']=7;  //G	Gly	Glycine
			valid_chars[(uint8_t) 'H']=8;  //H	His	Histidine
			valid_chars[(uint8_t) 'I']=9;  //I	Ile	Isoleucine
			valid_chars[(uint8_t) 'L']=10; //L	Leu	Leucine
			valid_chars[(uint8_t) 'K']=11; //K	Lys	Lysine
			valid_chars[(uint8_t) 'M']=12; //M	Met	Methionine
			valid_chars[(uint8_t) 'F']=13; //F	Phe	Phenylalanine
			valid_chars[(uint8_t) 'P']=14; //P	Pro	Proline
			valid_chars[(uint8_t) 'S']=15; //S	Ser	Serine
			valid_chars[(uint8_t) 'T']=16; //T	Thr	Threonine
			valid_chars[(uint8_t) 'W']=17; //W	Trp	Tryptophan
			valid_chars[(uint8_t) 'Y']=18; //Y	Tyr	Tyrosine
			valid_chars[(uint8_t) 'V']=19; //V	Val	Valine
			valid_chars[(uint8_t) 'B']=20; //B	Asx	Aspartic acid or Asparagine
			valid_chars[(uint8_t) 'Z']=21; //Z	Glx	Glutamine or Glutamic acid
			valid_chars[(uint8_t) 'X']=22; //X	Xaa	Any amino acid

			maptable_to_bin[(uint8_t) 'A']=0;  //A	Ala	Alanine
			maptable_to_bin[(uint8_t) 'R']=1;  //R	Arg	Arginine
			maptable_to_bin[(uint8_t) 'N']=2;  //N	Asn	Asparagine
			maptable_to_bin[(uint8_t) 'D']=3;  //D	Asp	Aspartic acid
			maptable_to_bin[(uint8_t) 'C']=4;  //C	Cys	Cysteine
			maptable_to_bin[(uint8_t) 'Q']=5;  //Q	Gln	Glutamine
			maptable_to_bin[(uint8_t) 'E']=6;  //E	Glu	Glutamic acid
			maptable_to_bin[(uint8_t) 'G']=7;  //G	Gly	Glycine
			maptable_to_bin[(uint8_t) 'H']=8;  //H	His	Histidine
			maptable_to_bin[(uint8_t) 'I']=9;  //I	Ile	Isoleucine
			maptable_to_bin[(uint8_t) 'L']=10; //L	Leu	Leucine
			maptable_to_bin[(uint8_t) 'K']=11; //K	Lys	Lysine
			maptable_to_bin[(uint8_t) 'M']=12; //M	Met	Methionine
			maptable_to_bin[(uint8_t) 'F']=13; //F	Phe	Phenylalanine
			maptable_to_bin[(uint8_t) 'P']=14; //P	Pro	Proline
			maptable_to_bin[(uint8_t) 'S']=15; //S	Ser	Serine
			maptable_to_bin[(uint8_t) 'T']=16; //T	Thr	Threonine
			maptable_to_bin[(uint8_t) 'W']=17; //W	Trp	Tryptophan
			maptable_to_bin[(uint8_t) 'Y']=18; //Y	Tyr	Tyrosine
			maptable_to_bin[(uint8_t) 'V']=19; //V	Val	Valine
			maptable_to_bin[(uint8_t) 'B']=20; //B	Asx	Aspartic acid or Asparagine
			maptable_to_bin[(uint8_t) 'Z']=21; //Z	Glx	Glutamine or Glutamic acid
			maptable_to_bin[(uint8_t) 'X']=22; //X	Xaa	Any amino acid

			maptable_to_char[0]=(uint8_t) 'A';  //A	Ala	Alanine
			maptable_to_char[1]=(uint8_t) 'R';  //R	Arg	Arginine
			maptable_to_char[2]=(uint8_t) 'N';  //N	Asn	Asparagine
			maptable_to_char[3]=(uint8_t) 'D';  //D	Asp	Aspartic acid
			maptable_to_char[4]=(uint8_t) 'C';  //C	Cys	Cysteine
			maptable_to_char[5]=(uint8_t) 'Q';  //Q	Gln	Glutamine
			maptable_to_char[6]=(uint8_t) 'E';  //E	Glu	Glutamic acid
			maptable_to_char[7]=(uint8_t) 'G';  //G	Gly	Glycine
			maptable_to_char[8]=(uint8_t) 'H';  //H	His	Histidine
			maptable_to_char[9]=(uint8_t) 'I';  //I	Ile	Isoleucine
			maptable_to_char[10]=(uint8_t) 'L'; //L	Leu	Leucine
			maptable_to_char[11]=(uint8_t) 'K'; //K	Lys	Lysine
			maptable_to_char[12]=(uint8_t) 'M'; //M	Met	Methionine
			maptable_to_char[13]=(uint8_t) 'F'; //F	Phe	Phenylalanine
			maptable_to_char[14]=(uint8_t) 'P'; //P	Pro	Proline
			maptable_to_char[15]=(uint8_t) 'S'; //S	Ser	Serine
			maptable_to_char[16]=(uint8_t) 'T'; //T	Thr	Threonine
			maptable_to_char[17]=(uint8_t) 'W'; //W	Trp	Tryptophan
			maptable_to_char[18]=(uint8_t) 'Y'; //Y	Tyr	Tyrosine
			maptable_to_char[19]=(uint8_t) 'V'; //V	Val	Valine
			maptable_to_char[20]=(uint8_t) 'B'; //B	Asx	Aspartic acid or Asparagine
			maptable_to_char[21]=(uint8_t) 'Z'; //Z	Glx	Glutamine or Glutamic acid
			maptable_to_char[22]=(uint8_t) 'X'; //X	Xaa	Any amino acid
		default:
			break; //leave uninitialised
	};
}

void CAlphabet::clear_histogram()
{
	memset(histogram, 0, sizeof(histogram));
    print_histogram();
}

void CAlphabet::add_string_to_histogram(uint8_t* p, LONG len)
{
	for (LONG i=0; i<len; i++)
		add_byte_to_histogram(p[i]);
}

void CAlphabet::add_string_to_histogram(char* p, LONG len)
{
	for (LONG i=0; i<len; i++)
		add_byte_to_histogram(p[i]);
}

void CAlphabet::add_string_to_histogram(uint16_t* p, LONG len)
{
	SG_WARNING("computing byte histogram over word strings\n");
	uint8_t* b= (uint8_t*) p;
	for (LONG i=0; i<((LONG) sizeof(uint16_t))*len; i++)
		add_byte_to_histogram(b[i]);
}

void CAlphabet::add_string_to_histogram(SHORT* p, LONG len)
{
	SG_WARNING("computing byte histogram over word strings\n");
	uint8_t* b= (uint8_t*) p;
	for (LONG i=0; i<((LONG) sizeof(SHORT))*len; i++)
		add_byte_to_histogram(b[i]);
}

void CAlphabet::add_string_to_histogram(int32_t* p, LONG len)
{
	SG_WARNING("computing byte histogram over word strings\n");
	uint8_t* b= (uint8_t*) p;
	for (LONG i=0; i<((LONG) sizeof(int32_t))*len; i++)
		add_byte_to_histogram(b[i]);
}

void CAlphabet::add_string_to_histogram(uint32_t* p, LONG len)
{
	SG_WARNING("computing byte histogram over word strings\n");
	uint8_t* b= (uint8_t*) p;
	for (LONG i=0; i<((LONG) sizeof(uint32_t))*len; i++)
		add_byte_to_histogram(b[i]);
}

void CAlphabet::add_string_to_histogram(LONG* p, LONG len)
{
	SG_WARNING("computing byte histogram over word strings\n");
	uint8_t* b= (uint8_t*) p;
	for (LONG i=0; i<((LONG) sizeof(LONG))*len; i++)
		add_byte_to_histogram(b[i]);
}

void CAlphabet::add_string_to_histogram(ULONG* p, LONG len)
{
	SG_WARNING("computing byte histogram over word strings\n");
	uint8_t* b= (uint8_t*) p;
	for (LONG i=0; i<((LONG) sizeof(ULONG))*len; i++)
		add_byte_to_histogram(b[i]);
}

int32_t CAlphabet::get_max_value_in_histogram()
{
	int32_t max_sym=-1;
	for (int32_t i=(int32_t) (1 <<(sizeof(uint8_t)*8))-1;i>=0; i--)
	{
		if (histogram[i])
		{
			max_sym=i;
			break;
		}
	}

	return max_sym;
}

int32_t CAlphabet::get_num_symbols_in_histogram()
{
	int32_t num_sym=0;
	for (int32_t i=0; i<(int32_t) (1 <<(sizeof(uint8_t)*8)); i++)
	{
		if (histogram[i])
			num_sym++;
	}

	return num_sym;
}

int32_t CAlphabet::get_num_bits_in_histogram()
{
	int32_t num_sym=get_num_symbols_in_histogram();
	if (num_sym>0)
		return (int32_t) ceil(log((double) num_sym)/log((double) 2));
	else
		return 0;
}

void CAlphabet::print_histogram()
{
	for (int32_t i=0; i<(int32_t) (1 <<(sizeof(uint8_t)*8)); i++)
	{
		if (histogram[i])
			SG_PRINT( "hist[%d]=%lld\n", i, histogram[i]);
	}
}

bool CAlphabet::check_alphabet(bool print_error)
{
	bool result = true;

	for (int32_t i=0; i<(int32_t) (1 <<(sizeof(uint8_t)*8)); i++)
	{
		if (histogram[i]>0 && valid_chars[i]==0)
		{
			result=false;
			break;
		}
	}

	if (!result && print_error)
	{
		print_histogram();
      SG_ERROR( "ALPHABET does not contain all symbols in histogram\n");
	}

	return result;
}

bool CAlphabet::check_alphabet_size(bool print_error)
{
	if (get_num_bits_in_histogram() > get_num_bits())
	{
		if (print_error)
		{
			print_histogram();
			fprintf(stderr, "get_num_bits_in_histogram()=%i > get_num_bits()=%i\n", get_num_bits_in_histogram(), get_num_bits()) ;
         SG_ERROR( "ALPHABET too small to contain all symbols in histogram\n");
		}
		return false;
	}
	else
		return true;

}

void CAlphabet::copy_histogram(CAlphabet* a)
{
	memcpy(histogram, a->get_histogram(), sizeof(histogram));
}

const char* CAlphabet::get_alphabet_name(EAlphabet alphabet)
{
	
	int32_t idx;
	switch (alphabet)
	{
		case DNA:
			idx=0;
			break;
		case RAWDNA:
			idx=1;
			break;
		case RNA:
			idx=2;
			break;
		case PROTEIN:
			idx=3;
			break;
		case ALPHANUM:
			idx=4;
			break;
		case CUBE:
			idx=5;
			break;
		case RAWBYTE:
			idx=6;
			break;
		case IUPAC_NUCLEIC_ACID:
			idx=7;
			break;
		case IUPAC_AMINO_ACID:
			idx=8;
			break;
		case NONE:
			idx=9;
			break;
		default:
			idx=10;
			break;
	}
	return alphabet_names[idx];
}
