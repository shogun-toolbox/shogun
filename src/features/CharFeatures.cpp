#include "features/CharFeatures.h"
#include "lib/common.h"
#include "lib/File.h"

#define MAPTABLE_UNDEF ((1<<(8*sizeof(CHAR)))-1)

//define numbers for the bases 
const unsigned char CCharFeatures::B_A=0;
const unsigned char CCharFeatures::B_C=1;
const unsigned char CCharFeatures::B_G=2;
const unsigned char CCharFeatures::B_T=3;
const unsigned char CCharFeatures::B_N=4;
const unsigned char CCharFeatures::B_n=5;

CCharFeatures::CCharFeatures(E_OBS_ALPHABET a, long size) : CSimpleFeatures<CHAR>(size)
{
	alphabet_type=a;
	init_map_table();
}

CCharFeatures::CCharFeatures(const CCharFeatures & orig) : CSimpleFeatures<CHAR>(orig)
{
}

CCharFeatures::CCharFeatures(char* fname) : CSimpleFeatures<CHAR>(fname)
{
}

CFeatures* CCharFeatures::duplicate() const
{
	return new CCharFeatures(*this);
}


bool CCharFeatures::load(char* fname)
{
    long length=-1;
	long linelen=-1;
	long columns=-1;

	CFile f(fname, 'r', F_CHAR);
	feature_matrix=f.load_char_data(NULL, length);

    if (f.is_ok())
	{
		for (linelen=0; linelen<length; linelen++)
		{
			if (feature_matrix[linelen]=='\n')
			{
				num_features=linelen-1;
				break;
			}
		}

		num_vectors=length/linelen;

		if (length && (num_vectors*linelen==length))
		{
			for (int lines=0; lines<num_vectors; lines++)
			{
				for (columns=0; columns<num_features; columns++)
					feature_matrix[lines*num_features+columns]=feature_matrix[lines*linelen+columns];

				if (feature_matrix[lines*linelen+columns]!='\n')
				{
					CIO::message("line %d in file \"%s\" is corrupt\n", lines, fname);
					return false;
				}
			}

			return true;
		}
		else
			CIO::message("file is of zero size or no rectangular featurematrix of type CHAR\n");
	}

	return false;
}

bool CCharFeatures::save(char* fname)
{
	return false;
}

void CCharFeatures::init_map_table()
{
  int i;
  for (i=0; i<(1<<(8*sizeof(CHAR))); i++)
    maptable[i] = MAPTABLE_UNDEF ;

  switch (alphabet_type)
    {
	case CUBE:
	    maptable['1']=0;
	    maptable['2']=1;
	    maptable['3']=2;
	    maptable['4']=3;	
	    maptable['5']=4;	
	    maptable['6']=5;	//Translation '123456' -> 012345

	    maptable[0]='1';
	    maptable[1]='2';
	    maptable[2]='3';
	    maptable[3]='4';
	    maptable[4]='5';
	    maptable[5]='6';	//Translation 012345->'123456'

	    break;
    case PROTEIN:
      {
	int skip=0 ;
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
    case DNA:
    default:
	maptable['A']=B_A;
	maptable['C']=B_C;
	maptable['G']=B_G;
	maptable['T']=B_T;	
	maptable['N']=B_N;	
	maptable['n']=B_n;	//Translation ACGTNn -> 012345

	maptable[B_A]='A';
	maptable[B_C]='C';
	maptable[B_G]='G';
	maptable[B_T]='T';
	maptable[B_N]='N';
	maptable[B_n]='n';	//Translation 012345->ACGTNn
      break;

    };
}
