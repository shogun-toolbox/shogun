// Observation.cpp: implementation of the CObservation class.
//
//////////////////////////////////////////////////////////////////////

#include "hmm/Observation.h"
#include "lib/Mathmatics.h"
#include "lib/io.h"

#include <stdio.h>

const int MAPTABLE_LENGTH = 1 << (sizeof(T_MAPTABLE)*8) ;
T_OBSERVATIONS CObservation::maptable[MAPTABLE_LENGTH];

#define MAPTABLE_UNDEF ((1<<(8*sizeof(T_OBSERVATIONS)))-1)

//define numbers for the bases 
const unsigned char B_A=0;
const unsigned char B_C=1;
const unsigned char B_G=2;
const unsigned char B_T=3;
const unsigned char B_N=4;
const unsigned char B_n=5;


CObservation::CObservation(E_OBS_TYPE type, E_OBS_ALPHABET alphabet, int MAX_M_, int M_, int ORDER_)
{
	full_observation=NULL;
	observations=NULL;
	observation_len=NULL;
	label=NULL;

	this->observation_type=type;
	this->alphabet_type=alphabet;
	this->MAX_M= MAX_M_;
	this->M=M_;
	this->ORDER=ORDER_;
	this->max_T= -1;

	CIO::message("MM:%d\n",MAX_M);
	cleanup();

	init_map_table();
}

CObservation::CObservation(CObservation* pos, CObservation* neg)
{
  /*	sv_idx=-1;
	sv_num=0;*/
	this->full_observation=NULL;
	this->MAX_M=-1;
	this->M=-1;
	this->ORDER=1;

	if ( pos->get_alphabet() == neg->get_alphabet() )
	{
		MAX_M=math.min(pos->get_max_M(), neg->get_max_M());
	CIO::message("MM:%d\n",MAX_M);
		M=math.min(pos->get_M(), neg->get_M());
		ORDER=1;

		REALDIMENSION=pos->get_DIMENSION()+neg->get_DIMENSION();
		DIMENSION=REALDIMENSION;

		observations=new T_OBSERVATIONS*[DIMENSION];
		observation_len= new int[DIMENSION];
		label= new int[DIMENSION];

		if (pos->get_type() == POSTRAIN || pos->get_type() == NEGTRAIN)
			observation_type=TRAIN;
		else if (pos->get_type() == POSTEST || pos->get_type() == NEGTEST)
			observation_type=TEST;
		else
			observation_type=UNLABELED;

		for (int i=0; i<DIMENSION; i++)
		{
			if (i<pos->get_DIMENSION())
			{
				observations[i]=pos->get_obs_vector(i);
				observation_len[i]=pos->get_obs_T(i);
				label[i]=+1;
			}
			else
			{
				observations[i]=neg->get_obs_vector(i-pos->get_DIMENSION());
				observation_len[i]=neg->get_obs_T(i-pos->get_DIMENSION());
				label[i]=-1;
			}
		}

		this->max_T= math.max(pos->get_obs_max_T(), neg->get_obs_max_T());
	}
	else
	{
		observations=NULL;
		observation_len=NULL;
		label=NULL;

		this->max_T= -1;
		this->observation_type=NONE;
		this->alphabet_type=DNA;

		cleanup();
	}
}

CObservation::CObservation(FILE* file, E_OBS_TYPE type, E_OBS_ALPHABET alphabet, int MAX_M_, int M_, int ORDER_, int start, int width)
{
	full_observation=NULL;
	observations=NULL;
	observation_len=NULL;
	label=NULL;

	cleanup();
	load_observations(file,type,alphabet,MAX_M_,M_,ORDER_, start, width);
}

CObservation::~CObservation()
{
	cleanup();
}

void CObservation::cleanup()
{
	REALDIMENSION=-1;
	DIMENSION=-1;

	delete[] full_observation;
	full_observation=NULL;

	delete[] observations;
	observations=NULL;

	delete[] observation_len;
	observation_len=NULL;

	delete[] label;
	label=NULL;
}

/* read the whole observation sequence into memory

  -format specs: in_file (gene.lin)
		[AGCT]+<<EOF>>
*/
bool CObservation::load_observations(FILE* file, E_OBS_TYPE type, E_OBS_ALPHABET alphabet, int MAX_M_, int M_, int ORDER_, int start, int width)
{

	this->observation_type=type;
	this->alphabet_type=alphabet;
	this->MAX_M= MAX_M_;
	this->M=M_;
	this->ORDER=ORDER_;

	CIO::message("MM:%d\n",MAX_M);
	cleanup();
	init_map_table();

    int length=-1;
    int numread=0;

    if (!fseek(file, 0, SEEK_END))
    {
		if ((length=(int)ftell(file)) != -1)
		{
#warning fixme: depends on alphabet
			char* full_content = new char[length];
			full_observation = new T_OBSERVATIONS[length];

			rewind(file);
			numread=(int)fread(full_content, sizeof (unsigned char), length, file);

			if (numread==length)
			{
				int i;
				for (i=length-1; i>=0; i--)
				{
					full_observation[i]=full_content[i];
				}

				delete[] full_content;
				full_content=NULL;

				// count lines
				int lines = 0 ;
				for (i=0;i<length; i++)
				{
					if (full_observation[i]==(T_OBSERVATIONS) '\n')
						lines++ ;
				}

				observations=new T_OBSERVATIONS*[lines];
				observation_len=new int[lines];

				REALDIMENSION=lines ;
				DIMENSION=lines ;

				printf("found %i sequences\nallocating memory...", lines) ;


				// count letters per line
				int line=0 ;
				int time=0 ; 
				max_T=-1;

				for (i=0;i<length; i++)
				{
					if (full_observation[i]==(T_OBSERVATIONS) '\n')
					{
						observations[line]=(T_OBSERVATIONS*) full_observation+i-time;
						observation_len[line]= (width>0) ? math.min(time,width) : time;
						full_observation[i]=(T_OBSERVATIONS) '\0';

						if (translate_from_single_order(observations[line], time, start) < 0)
							CIO::message(stderr,"wrong character(s) in line %i\n", line);

						if (time>max_T)
							max_T=time;

						line++;
						time=0;
					} 
					else time++;
				}

				CIO::message("done\n") ;
				CIO::message("maximum length %i (start: %d width: %d)\n", max_T, start, width);
			}
			else
				CIO::message("error reading file\n");
		}
    }
    return numread==length;
}

int CObservation::get_alphabet_size(E_OBS_ALPHABET a)
{
	switch (a)
	{
		case CUBE:
			return 6;
		case PROTEIN:
			return 21;
		case ALPHANUM:
			return 36;
		case DNA:
			return 4;
		default:
			return -1;
	};
}

//init map_table
void CObservation::init_map_table()
{
  int i;
  for (i=0; i<(1<<(8*sizeof(T_OBSERVATIONS))); i++)
    maptable[i] = MAPTABLE_UNDEF ;

  switch (alphabet_type)
    {
	case CUBE:
	    maptable[(unsigned char) '1']=0;
	    maptable[(unsigned char) '2']=1;
	    maptable[(unsigned char) '3']=2;
	    maptable[(unsigned char) '4']=3;	
	    maptable[(unsigned char) '5']=4;	
	    maptable[(unsigned char) '6']=5;	//Translation '123456' -> 012345

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
	maptable[(unsigned char) 'A']=B_A;
	maptable[(unsigned char) 'C']=B_C;
	maptable[(unsigned char) 'G']=B_G;
	maptable[(unsigned char) 'T']=B_T;	
	maptable[(unsigned char) 'N']=B_N;	
	maptable[(unsigned char) 'n']=B_n;	//Translation ACGTNn -> 012345

	maptable[B_A]='A';
	maptable[B_C]='C';
	maptable[B_G]='G';
	maptable[B_T]='T';
	maptable[B_N]='N';
	maptable[B_n]='n';	//Translation 012345->ACGTNn
      break;

    };
}

//translate ACGT <-> 0123 in observations
int CObservation::translate_from_single_order(T_OBSERVATIONS* observations_, int sequence_length, int start)
{
	int i,j, fac=1 ;
	T_OBSERVATIONS value=0;

	for (i=sequence_length-1; i>= ((int) ORDER)-1; i--)	//convert interval of size T
	{
		value=0;
		for (j=i; j>=i-((int) ORDER)+1; j--)
		{
			if ((maptable[observations_[j]]>=M) || (maptable[observations_[j]]==MAPTABLE_UNDEF))
			{
				CIO::message(stderr,"wrong: %c -> %i,%i,%i,%i\n",observations_[j],(int)maptable[observations_[j]],j,i,sequence_length-1) ;
				fac=-1 ;
			}
			value= (value >> MAX_M) | (maptable[observations_[j]] << (MAX_M * (ORDER-1)));
		}
		observations_[i]=value;
	}

	for (i=ORDER-2;i>=0;i--)
	{
		value=0;
		for (j=i; j>=i-ORDER+1; j--)
		{
			value= (value >> MAX_M);
			if (j>=0)
				value|=maptable[observations_[j]] << (MAX_M * (ORDER-1));
		}
		observations_[i]=value;
	}

	for (i=start; i<sequence_length; i++)	
	{
		observations_[i-start]=observations_[i];
		//if (observations_[i-start]>16)
			//CIO::message("%d,%d : %d\n", MAX_M, ORDER, observations_[i-start]);
	}

	return fac*(sequence_length-(ORDER-1)) ;
}

//translate ACGT <-> 0123 in observations
void CObservation::translate_to_single_order(T_OBSERVATIONS* observations_, int sequence_length)
{
	T_OBSERVATIONS mask=(1 << MAX_M ) - 1;

	for (int i=0; i<sequence_length; i++)
		observations_[i]=maptable[mask & observations_[i]];
}

