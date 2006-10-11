/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Soeren Sonnenburg
 * Copyright (C) 2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <string.h>
#include <math.h>

#include "features/Alphabet.h"
#include "lib/io.h"

//define numbers for the bases 
const BYTE CAlphabet::B_A=0;
const BYTE CAlphabet::B_C=1;
const BYTE CAlphabet::B_G=2;
const BYTE CAlphabet::B_T=3;
const BYTE CAlphabet::MAPTABLE_UNDEF=0xff;
const CHAR* CAlphabet::alphabet_names[]={"DNA", "PROTEIN", "ALPHANUM", "CUBE", "RAW", "IUPAC_NUCLEIC_ACID", "IUPAC_AMINO_ACID", "NONE", "UNKNOWN"};

CAlphabet::CAlphabet(CHAR* al, INT len)
{
	E_ALPHABET alpha=NONE;

	if (len>=(INT) strlen("DNA") && !strncmp(al, "DNA", strlen("DNA")))
		alpha = DNA;
	else if (len>=(INT) strlen("PROTEIN") && !strncmp(al, "PROTEIN", strlen("PROTEIN")))
		alpha = PROTEIN;
	else if (len>=(INT) strlen("ALPHANUM") && !strncmp(al, "ALPHANUM", strlen("ALPHANUM")))
		alpha = ALPHANUM;
	else if (len>=(INT) strlen("CUBE") && !strncmp(al, "CUBE", strlen("CUBE")))
		alpha = CUBE;
	else if (len>=(INT) strlen("BYTE") && !strncmp(al, "BYTE", strlen("BYTE")) || 
			(len>=(INT) strlen("RAW") && !strncmp(al, "RAW", strlen("RAW"))))
		alpha = RAWBYTE;
	else if (len>=(INT) strlen("IUPAC_NUCLEIC_ACID") && !strncmp(al, "IUPAC_NUCLEIC_ACID", strlen("IUPAC_NUCLEIC_ACID")))
		alpha = IUPAC_NUCLEIC_ACID;
	else if (len>=(INT) strlen("IUPAC_AMINO_ACID") && !strncmp(al, "IUPAC_AMINO_ACID", strlen("IUPAC_AMINO_ACID")))
		alpha = IUPAC_AMINO_ACID;
	else
		CIO::message(M_ERROR, "unknown alphabet %s\n", al);
	
	set_alphabet(alpha);
}

CAlphabet::CAlphabet(E_ALPHABET alpha)
{
	set_alphabet(alpha);
}

CAlphabet::CAlphabet(CAlphabet* a)
{
	set_alphabet(a->get_alphabet());
}

CAlphabet::~CAlphabet()
{
}

bool CAlphabet::set_alphabet(E_ALPHABET alpha)
{
	bool result=true;
	alphabet=alpha;

	switch (alphabet)
	{
		case DNA:
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

	num_bits=(INT) ceil(log((double) num_symbols)/log((double) 2));
	init_map_table();

	CIO::message(M_DEBUG, "initialised alphabet %s\n", get_alphabet_name(alphabet));

	return result;
}

void CAlphabet::init_map_table()
{
	INT i;
	for (i=0; i<(1<<(8*sizeof(BYTE))); i++)
	{
		maptable_to_bin[i] = MAPTABLE_UNDEF;
		maptable_to_char[i] = MAPTABLE_UNDEF;
		valid_chars[i] = 0;
	}

	switch (alphabet)
	{
		case CUBE:
			valid_chars[(BYTE) '1']=1;
			valid_chars[(BYTE) '2']=1;
			valid_chars[(BYTE) '3']=1;
			valid_chars[(BYTE) '4']=1;	
			valid_chars[(BYTE) '5']=1;	
			valid_chars[(BYTE) '6']=1;	//Translation '123456' -> 012345

			maptable_to_bin[(BYTE) '1']=0;
			maptable_to_bin[(BYTE) '2']=1;
			maptable_to_bin[(BYTE) '3']=2;
			maptable_to_bin[(BYTE) '4']=3;	
			maptable_to_bin[(BYTE) '5']=4;	
			maptable_to_bin[(BYTE) '6']=5;	//Translation '123456' -> 012345

			maptable_to_char[(BYTE) 0]='1';
			maptable_to_char[(BYTE) 1]='2';
			maptable_to_char[(BYTE) 2]='3';
			maptable_to_char[(BYTE) 3]='4';
			maptable_to_char[(BYTE) 4]='5';
			maptable_to_char[(BYTE) 5]='6';	//Translation 012345->'123456'
			break;

		case PROTEIN:
			{
				INT skip=0 ;
				for (i=0; i<21; i++)
				{
					if (i==1) skip++ ;
					if (i==8) skip++ ;
					if (i==12) skip++ ;
					if (i==17) skip++ ;
					valid_chars['a'+i+skip]=1;
					maptable_to_bin['a'+i+skip]=i ;
					maptable_to_char[i]='a'+i+skip ;
				} ;                   //Translation 012345->acde...xy -- the protein code
			} ;
			break;

		case ALPHANUM:
			{
				for (i=0; i<26; i++)
				{
					valid_chars['a'+i]=1;
					maptable_to_bin['a'+i]=i ;
					maptable_to_char[i]='a'+i ;
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
					maptable_to_char[i]=i;
					maptable_to_char[i]=i;
				}
			}
			break;

		case DNA:
			valid_chars[(BYTE) 'A']=B_A;
			valid_chars[(BYTE) 'C']=B_C;
			valid_chars[(BYTE) 'G']=B_G;
			valid_chars[(BYTE) 'T']=B_T;	

			maptable_to_bin[(BYTE) 'A']=B_A;
			maptable_to_bin[(BYTE) 'C']=B_C;
			maptable_to_bin[(BYTE) 'G']=B_G;
			maptable_to_bin[(BYTE) 'T']=B_T;	

			maptable_to_char[B_A]='A';
			maptable_to_char[B_C]='C';
			maptable_to_char[B_G]='G';
			maptable_to_char[B_T]='T';
			break;

		case IUPAC_NUCLEIC_ACID:
			valid_chars[(BYTE) 'A']=1; // A	Adenine
			valid_chars[(BYTE) 'C']=1; // C	Cytosine
			valid_chars[(BYTE) 'G']=1; // G	Guanine
			valid_chars[(BYTE) 'T']=1; // T	Thymine
			valid_chars[(BYTE) 'U']=1; // U	Uracil
			valid_chars[(BYTE) 'R']=1; // R	Purine (A or G)
			valid_chars[(BYTE) 'Y']=1; // Y	Pyrimidine (C, T, or U)
			valid_chars[(BYTE) 'M']=1; // M	C or A
			valid_chars[(BYTE) 'K']=1; // K	T, U, or G
			valid_chars[(BYTE) 'W']=1; // W	T, U, or A
			valid_chars[(BYTE) 'S']=1; // S	C or G
			valid_chars[(BYTE) 'B']=1; // B	C, T, U, or G (not A)
			valid_chars[(BYTE) 'D']=1; // D	A, T, U, or G (not C)
			valid_chars[(BYTE) 'H']=1; // H	A, T, U, or C (not G)
			valid_chars[(BYTE) 'V']=1; // V	A, C, or G (not T, not U)
			valid_chars[(BYTE) 'N']=1; // N	Any base (A, C, G, T, or U)

			maptable_to_bin[(BYTE) 'A']=0; // A	Adenine
			maptable_to_bin[(BYTE) 'C']=1; // C	Cytosine
			maptable_to_bin[(BYTE) 'G']=2; // G	Guanine
			maptable_to_bin[(BYTE) 'T']=3; // T	Thymine
			maptable_to_bin[(BYTE) 'U']=4; // U	Uracil
			maptable_to_bin[(BYTE) 'R']=5; // R	Purine (A or G)
			maptable_to_bin[(BYTE) 'Y']=6; // Y	Pyrimidine (C, T, or U)
			maptable_to_bin[(BYTE) 'M']=7; // M	C or A
			maptable_to_bin[(BYTE) 'K']=8; // K	T, U, or G
			maptable_to_bin[(BYTE) 'W']=9; // W	T, U, or A
			maptable_to_bin[(BYTE) 'S']=10; // S	C or G
			maptable_to_bin[(BYTE) 'B']=11; // B	C, T, U, or G (not A)
			maptable_to_bin[(BYTE) 'D']=12; // D	A, T, U, or G (not C)
			maptable_to_bin[(BYTE) 'H']=13; // H	A, T, U, or C (not G)
			maptable_to_bin[(BYTE) 'V']=14; // V	A, C, or G (not T, not U)
			maptable_to_bin[(BYTE) 'N']=15; // N	Any base (A, C, G, T, or U)

			maptable_to_char[0]=(BYTE) 'A'; // A	Adenine
			maptable_to_char[1]=(BYTE) 'C'; // C	Cytosine
			maptable_to_char[2]=(BYTE) 'G'; // G	Guanine
			maptable_to_char[3]=(BYTE) 'T'; // T	Thymine
			maptable_to_char[4]=(BYTE) 'U'; // U	Uracil
			maptable_to_char[5]=(BYTE) 'R'; // R	Purine (A or G)
			maptable_to_char[6]=(BYTE) 'Y'; // Y	Pyrimidine (C, T, or U)
			maptable_to_char[7]=(BYTE) 'M'; // M	C or A
			maptable_to_char[8]=(BYTE) 'K'; // K	T, U, or G
			maptable_to_char[9]=(BYTE) 'W'; // W	T, U, or A
			maptable_to_char[10]=(BYTE) 'S'; // S	C or G
			maptable_to_char[11]=(BYTE) 'B'; // B	C, T, U, or G (not A)
			maptable_to_char[12]=(BYTE) 'D'; // D	A, T, U, or G (not C)
			maptable_to_char[13]=(BYTE) 'H'; // H	A, T, U, or C (not G)
			maptable_to_char[14]=(BYTE) 'V'; // V	A, C, or G (not T, not U)
			maptable_to_char[15]=(BYTE) 'N'; // N	Any base (A, C, G, T, or U)
			break;

		case IUPAC_AMINO_ACID:
			valid_chars[(BYTE) 'A']=0;  //A	Ala	Alanine
			valid_chars[(BYTE) 'R']=1;  //R	Arg	Arginine
			valid_chars[(BYTE) 'N']=2;  //N	Asn	Asparagine
			valid_chars[(BYTE) 'D']=3;  //D	Asp	Aspartic acid
			valid_chars[(BYTE) 'C']=4;  //C	Cys	Cysteine
			valid_chars[(BYTE) 'Q']=5;  //Q	Gln	Glutamine
			valid_chars[(BYTE) 'E']=6;  //E	Glu	Glutamic acid
			valid_chars[(BYTE) 'G']=7;  //G	Gly	Glycine
			valid_chars[(BYTE) 'H']=8;  //H	His	Histidine
			valid_chars[(BYTE) 'I']=9;  //I	Ile	Isoleucine
			valid_chars[(BYTE) 'L']=10; //L	Leu	Leucine
			valid_chars[(BYTE) 'K']=11; //K	Lys	Lysine
			valid_chars[(BYTE) 'M']=12; //M	Met	Methionine
			valid_chars[(BYTE) 'F']=13; //F	Phe	Phenylalanine
			valid_chars[(BYTE) 'P']=14; //P	Pro	Proline
			valid_chars[(BYTE) 'S']=15; //S	Ser	Serine
			valid_chars[(BYTE) 'T']=16; //T	Thr	Threonine
			valid_chars[(BYTE) 'W']=17; //W	Trp	Tryptophan
			valid_chars[(BYTE) 'Y']=18; //Y	Tyr	Tyrosine
			valid_chars[(BYTE) 'V']=19; //V	Val	Valine
			valid_chars[(BYTE) 'B']=20; //B	Asx	Aspartic acid or Asparagine
			valid_chars[(BYTE) 'Z']=21; //Z	Glx	Glutamine or Glutamic acid
			valid_chars[(BYTE) 'X']=22; //X	Xaa	Any amino acid

			maptable_to_bin[(BYTE) 'A']=0;  //A	Ala	Alanine
			maptable_to_bin[(BYTE) 'R']=1;  //R	Arg	Arginine
			maptable_to_bin[(BYTE) 'N']=2;  //N	Asn	Asparagine
			maptable_to_bin[(BYTE) 'D']=3;  //D	Asp	Aspartic acid
			maptable_to_bin[(BYTE) 'C']=4;  //C	Cys	Cysteine
			maptable_to_bin[(BYTE) 'Q']=5;  //Q	Gln	Glutamine
			maptable_to_bin[(BYTE) 'E']=6;  //E	Glu	Glutamic acid
			maptable_to_bin[(BYTE) 'G']=7;  //G	Gly	Glycine
			maptable_to_bin[(BYTE) 'H']=8;  //H	His	Histidine
			maptable_to_bin[(BYTE) 'I']=9;  //I	Ile	Isoleucine
			maptable_to_bin[(BYTE) 'L']=10; //L	Leu	Leucine
			maptable_to_bin[(BYTE) 'K']=11; //K	Lys	Lysine
			maptable_to_bin[(BYTE) 'M']=12; //M	Met	Methionine
			maptable_to_bin[(BYTE) 'F']=13; //F	Phe	Phenylalanine
			maptable_to_bin[(BYTE) 'P']=14; //P	Pro	Proline
			maptable_to_bin[(BYTE) 'S']=15; //S	Ser	Serine
			maptable_to_bin[(BYTE) 'T']=16; //T	Thr	Threonine
			maptable_to_bin[(BYTE) 'W']=17; //W	Trp	Tryptophan
			maptable_to_bin[(BYTE) 'Y']=18; //Y	Tyr	Tyrosine
			maptable_to_bin[(BYTE) 'V']=19; //V	Val	Valine
			maptable_to_bin[(BYTE) 'B']=20; //B	Asx	Aspartic acid or Asparagine
			maptable_to_bin[(BYTE) 'Z']=21; //Z	Glx	Glutamine or Glutamic acid
			maptable_to_bin[(BYTE) 'X']=22; //X	Xaa	Any amino acid

			maptable_to_char[0]=(BYTE) 'A';  //A	Ala	Alanine
			maptable_to_char[1]=(BYTE) 'R';  //R	Arg	Arginine
			maptable_to_char[2]=(BYTE) 'N';  //N	Asn	Asparagine
			maptable_to_char[3]=(BYTE) 'D';  //D	Asp	Aspartic acid
			maptable_to_char[4]=(BYTE) 'C';  //C	Cys	Cysteine
			maptable_to_char[5]=(BYTE) 'Q';  //Q	Gln	Glutamine
			maptable_to_char[6]=(BYTE) 'E';  //E	Glu	Glutamic acid
			maptable_to_char[7]=(BYTE) 'G';  //G	Gly	Glycine
			maptable_to_char[8]=(BYTE) 'H';  //H	His	Histidine
			maptable_to_char[9]=(BYTE) 'I';  //I	Ile	Isoleucine
			maptable_to_char[10]=(BYTE) 'L'; //L	Leu	Leucine
			maptable_to_char[11]=(BYTE) 'K'; //K	Lys	Lysine
			maptable_to_char[12]=(BYTE) 'M'; //M	Met	Methionine
			maptable_to_char[13]=(BYTE) 'F'; //F	Phe	Phenylalanine
			maptable_to_char[14]=(BYTE) 'P'; //P	Pro	Proline
			maptable_to_char[15]=(BYTE) 'S'; //S	Ser	Serine
			maptable_to_char[16]=(BYTE) 'T'; //T	Thr	Threonine
			maptable_to_char[17]=(BYTE) 'W'; //W	Trp	Tryptophan
			maptable_to_char[18]=(BYTE) 'Y'; //Y	Tyr	Tyrosine
			maptable_to_char[19]=(BYTE) 'V'; //V	Val	Valine
			maptable_to_char[20]=(BYTE) 'B'; //B	Asx	Aspartic acid or Asparagine
			maptable_to_char[21]=(BYTE) 'Z'; //Z	Glx	Glutamine or Glutamic acid
			maptable_to_char[22]=(BYTE) 'X'; //X	Xaa	Any amino acid
		default:
			break; //leave uninitialised
	};
}

void CAlphabet::clear_histogram()
{
	memset(histogram, 0, sizeof(LONG) * (1 << (sizeof(BYTE)*8)));
}

void CAlphabet::add_string_to_histogram(BYTE* p, INT len)
{
	for (INT i=0; i<len; i++)
		add_byte_to_histogram(p[i]);
}

void CAlphabet::add_string_to_histogram(CHAR* p, INT len)
{
	for (INT i=0; i<len; i++)
		add_byte_to_histogram(p[i]);
}

INT CAlphabet::get_max_value_in_histogram()
{
	INT max_sym=-1;
	for (INT i=(INT) (1 <<(sizeof(BYTE)*8))-1;i>=0; i--)
	{
		if (histogram[i])
		{
			max_sym=i;
			break;
		}
	}

	return max_sym;
}

INT CAlphabet::get_num_symbols_in_histogram()
{
	INT num_sym=0;
	for (INT i=0; i<(INT) (1 <<(sizeof(BYTE)*8)); i++)
	{
		if (histogram[i])
			num_sym++;
	}

	return num_sym;
}

INT CAlphabet::get_num_bits_in_histogram()
{
	INT num_sym=get_num_symbols_in_histogram();
	if (num_sym>0)
		return (INT) ceil(log((double) num_sym)/log((double) 2));
	else
		return 0;
}

void CAlphabet::print_histogram()
{
	for (INT i=0; i<(INT) (1 <<(sizeof(BYTE)*8)); i++)
	{
		if (histogram[i])
			CIO::message(M_MESSAGEONLY, "hist[%d]=%ld\n", i, histogram[i]);
	}
}

bool CAlphabet::check_alphabet(bool print_error)
{
	bool result = true;

	for (INT i=0; i<(INT) (1 <<(sizeof(BYTE)*8)); i++)
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
		CIO::message(M_ERROR, "ALPHABET does not contain all symbols in histogram\n");
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
			CIO::message(M_ERROR, "ALPHABET too small to contain all symbols in histogram\n");
		}
		return false;
	}
	else
		return true;

}

const CHAR* CAlphabet::get_alphabet_name(E_ALPHABET alphabet)
{
	
	INT idx;
	switch (alphabet)
	{
		case DNA:
			idx=0;
			break;
		case PROTEIN:
			idx=1;
			break;
		case ALPHANUM:
			idx=2;
			break;
		case CUBE:
			idx=3;
			break;
		case RAWBYTE:
			idx=4;
			break;
		case IUPAC_NUCLEIC_ACID:
			idx=5;
			break;
		case IUPAC_AMINO_ACID:
			idx=6;
			break;
		case NONE:
			idx=7;
			break;
		default:
			idx=8;
			break;
	}
	return alphabet_names[idx];
}
