/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/CharFeatures.h"
#include "lib/common.h"
#include "lib/File.h"

#define MAPTABLE_UNDEF ((1<<(8*sizeof(CHAR)))-1)

//define numbers for the bases 
const BYTE CCharFeatures::B_A=0;
const BYTE CCharFeatures::B_C=1;
const BYTE CCharFeatures::B_G=2;
const BYTE CCharFeatures::B_T=3;
const BYTE CCharFeatures::B_star=4;
const BYTE CCharFeatures::B_N=5;
const BYTE CCharFeatures::B_n=6;

CCharFeatures::CCharFeatures(E_ALPHABET a, LONG size) : CSimpleFeatures<CHAR>(size)
{
	alphabet_type=a;
	init_map_table();
}

CCharFeatures::CCharFeatures(const CCharFeatures & orig) : CSimpleFeatures<CHAR>(orig)
{
}

CCharFeatures::CCharFeatures(E_ALPHABET alphabet, CHAR* feature_matrix, INT num_feat, INT num_vec) : CSimpleFeatures<CHAR>(feature_matrix, num_feat, num_vec)
{
	alphabet_type=alphabet;
	init_map_table();
}

CCharFeatures::CCharFeatures(E_ALPHABET a, CHAR* fname) : CSimpleFeatures<CHAR>(fname)
{
	alphabet_type=a;
	init_map_table();
	load(fname);
}

CFeatures* CCharFeatures::duplicate() const
{
	return new CCharFeatures(*this);
}


bool CCharFeatures::load(CHAR* fname)
{
	CIO::message(M_INFO, "loading...\n");
    LONG length=0;
	LONG linelen=0;

	CFile f(fname, 'r', F_CHAR);
	feature_matrix=f.load_char_data(NULL, length);

    if (f.is_ok())
	{
		for (linelen=0; linelen<length; linelen++)
		{
			if (feature_matrix[linelen]=='\n')
			{
				num_features=linelen;
				linelen++;
				break;
			}
		}

		num_vectors=length/linelen;

		CIO::message(M_INFO, "file contains %ldx%ld vectors x features\n", num_vectors, num_features);

		if (length && (num_vectors*linelen==length))
		{
			for (INT lines=0; lines<num_vectors; lines++)
			{
				for (INT columns=0; columns<num_features; columns++)
					feature_matrix[lines*num_features+columns]=feature_matrix[lines*linelen+columns];

				if (feature_matrix[lines*linelen+num_features]!='\n')
				{
					CIO::message(M_ERROR, "line %d in file \"%s\" is corrupt\n", lines, fname);
					return false;
				}
			}

			return true;
		}
		else
			CIO::message(M_ERROR, "file is of zero size or no rectangular featurematrix of type CHAR\n");
	}
	else
		CIO::message(M_ERROR, "reading file failed\n");

	return false;
}

bool CCharFeatures::save(CHAR* fname)
{
	return false;
}

void CCharFeatures::init_map_table()
{
  INT i;
  for (i=0; i<(1<<(8*sizeof(CHAR))); i++)
    maptable[i] = MAPTABLE_UNDEF ;

  switch (alphabet_type)
  {
      case CUBE:
          maptable[(BYTE) '1']=0;
          maptable[(BYTE) '2']=1;
          maptable[(BYTE) '3']=2;
          maptable[(BYTE) '4']=3;	
          maptable[(BYTE) '5']=4;	
          maptable[(BYTE) '6']=5;	//Translation '123456' -> 012345

          maptable[(BYTE) 0]='1';
          maptable[(BYTE) 1]='2';
          maptable[(BYTE) 2]='3';
          maptable[(BYTE) 3]='4';
          maptable[(BYTE) 4]='5';
          maptable[(BYTE) 5]='6';	//Translation 012345->'123456'

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
                  maptable[i]='a'+i+skip ;
                  maptable['a'+i+skip]=i ;
                  //printf("maptable[%c]=%i\n", 'a'+i+skip, i) ;
              } ;                   //Translation 012345->acde...xy -- the protein code
          } ;
          break;
      case ALPHANUM:
          {
              for (i=0; i<26; i++)
              {
                  maptable[i]='a'+i ;
                  maptable['a'+i]=i ;
              } ;
              for (i=0; i<10; i++)
              {
                  maptable[26+i]='0'+i ;
                  maptable['0'+i]=26+i ;
              } ;        //Translation 012345->acde...xy0123456789
          } ;
          break;
      case BYTE:
          {
              //identity
              for (i=0; i<256; i++)
                  maptable[i]=i;
          }
      case DNA:
      default:
          maptable[(BYTE) 'A']=B_A;
          maptable[(BYTE) 'C']=B_C;
          maptable[(BYTE) 'G']=B_G;
          maptable[(BYTE) 'T']=B_T;	
          maptable[(BYTE) '*']=B_star;	
          maptable[(BYTE) 'N']=B_N;	
          maptable[(BYTE) 'n']=B_n;	//Translation ACGTNn -> 012345

          maptable[B_A]='A';
          maptable[B_C]='C';
          maptable[B_G]='G';
          maptable[B_T]='T';
          maptable[B_star]='*';
          maptable[B_N]='N';
          maptable[B_n]='n';	//Translation 012345->ACGTNn
          break;
  };
}
