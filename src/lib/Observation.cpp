// Observation.cpp: implementation of the CObservation class.
//
//////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include "lib/Observation.h"
#include "lib/Mathmatics.h"

const int MAPTABLE_LENGTH = 1 << (sizeof(T_MAPTABLE)*8) ;
T_OBSERVATIONS CObservation::maptable[MAPTABLE_LENGTH];

#define MAPTABLE_UNDEF ((1<<(8*sizeof(T_OBSERVATIONS)))-1)

CObservation::CObservation(E_OBS_TYPE type, E_OBS_ALPHABET alphabet, int MAX_M, int M, int ORDER)
{
	full_content=NULL;
	observations=NULL;
	observation_len=NULL;
	label=NULL;

	this->observation_type=type;
	this->alphabet_type=alphabet;
	this->MAX_M= MAX_M;
	this->M=M;
	this->ORDER=ORDER;
	this->max_T= -1;

	cleanup();

	init_map_table();
}

CObservation::CObservation(CObservation* pos, CObservation* neg)
{
	sv_idx=-1;
	sv_num=0;
	this->full_content=NULL;
	this->MAX_M=-1;
	this->M=-1;
	this->ORDER=1;

	if ( pos->get_alphabet() == neg->get_alphabet() )
	{
		MAX_M=math.min(pos->get_max_M(), neg->get_max_M());
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

CObservation::CObservation(FILE* file, E_OBS_TYPE type, E_OBS_ALPHABET alphabet, int MAX_M, int M, int ORDER)
{
	full_content=NULL;
	observations=NULL;
	observation_len=NULL;
	label=NULL;

	cleanup();
	load_observations(file,type,alphabet,MAX_M,M,ORDER);
}

CObservation::~CObservation()
{
	cleanup();
}

void CObservation::cleanup()
{
	sv_idx=-1;
	sv_num=0;

	REALDIMENSION=-1;
	DIMENSION=-1;

	delete[] full_content;
	full_content=NULL;

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
bool CObservation::load_observations(FILE* file, E_OBS_TYPE type, E_OBS_ALPHABET alphabet, int MAX_M, int M, int ORDER)
{

	this->observation_type=type;
	this->alphabet_type=alphabet;
	this->MAX_M= MAX_M;
	this->M=M;
	this->ORDER=ORDER;

	cleanup();
	init_map_table();

    int length=-1;
    int numread=0;

    if (!fseek(file, 0, SEEK_END))
    {
		if ((length=(int)ftell(file)) != -1)
		{
			full_content = new char[length];
			
			rewind(file);
			numread=(int)fread(full_content, sizeof (unsigned char), length, file);

			// count lines
			int i, lines = 0 ;

			for (i=0;i<length; i++)
			{
				if (full_content[i]=='\n')
					lines++ ;
			}

			observations=new T_OBSERVATIONS*[lines];
			observation_len=new int[lines];

			REALDIMENSION=lines ;
			DIMENSION=lines ;

			printf("found %i sequences\nallocating memory...", lines) ;
			fflush(stdout);

			// count letters per line
			int line=0 ;
			int time=0 ; 
			max_T=-1;

			for (i=0;i<length; i++)
			{
				if (full_content[i]=='\n')
				{
					observations[line]=(T_OBSERVATIONS*) full_content+i-time;
					observation_len[line]=time;
					full_content[i]='\0';

					if (translate_from_single_order(observations[line], time) < 0)
						fprintf(stderr,"wrong character(s) in line %i\n", line) ;

					if (time>max_T)
						max_T=time;

					line++ ;
					time=0 ;
				} 
				else time++ ;
			} ;

			printf("done\n") ;
			printf("maximum length %i \n", max_T) ;
			fflush(stdout);

		}
    }
    return numread==length;
}

bool CObservation::add_support_vectors(FILE* file, int num_sv)
{
    bool result=false;
    T_OBSERVATIONS** obs=new T_OBSERVATIONS*[DIMENSION+num_sv];
    int* obs_len= new int[DIMENSION+num_sv];
    int* lab= new int[DIMENSION+num_sv];
    int max_T_SVM=0;
    int i=0;

    for (i=0; i<DIMENSION; i++)
    {
	    obs[i]=get_obs_vector(i);
	    obs_len[i]=get_obs_T(i);
	    lab[i]=get_label(i);
    }
    
    delete[] observations;
    delete[] observation_len;
    delete[] label;
    
    observations=obs;
    observation_len=obs_len;
    label=lab;

    char character;
    for (i=DIMENSION; i<DIMENSION+num_sv; i++)
    {
	while ((fread(&character, sizeof(char), 1, file))==1 && character !=':');

	if (character==':')
	{
	    int filepos=ftell(file);
	    int line_length=-1;
	    char* line=NULL;

	    while ((fread(&character, sizeof(char), 1, file))==1 && character!='\n' );

	    if (character=='\n')
	    {
		line_length=ftell(file)-filepos-1;
		line= new char[line_length];
		fseek(file, filepos, SEEK_SET);
		fread(line, sizeof(char), line_length, file);

		observations[i]=(T_OBSERVATIONS*) line;
		observation_len[i]=line_length;
		label[i]=0;

		translate_from_single_order(observations[i], observation_len[i]);
#ifdef DEBUG
		printf(">");
		for (int j=0; j<line_length; j++)
		    printf("%d",(int) observations[i][j]);
		printf("<\n");
#endif
		if (line_length>max_T_SVM)
		    max_T_SVM=line_length;
	    }
	    else 
		return false;
	}
	else 
	    return false;
    }
    
    if (max_T < max_T_SVM)
	max_T=max_T_SVM;

    printf("read %d support vectors\n", i-DIMENSION);

    sv_num=num_sv;
    sv_idx=DIMENSION;
    //DIMENSION+=num_sv; // we do not want the sv's to count

    return result;
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

//translate ACGT <-> 0123 in observations
int CObservation::translate_from_single_order(T_OBSERVATIONS* observations_, int sequence_length)
{
  int i,j, fac=1 ;
  T_OBSERVATIONS value=0;

  for (i=sequence_length-1; i>= ORDER-1; i--)	//convert interval of size T
    {
      value=0;
      for (j=i; j>=i-ORDER+1; j--)
	{
	  if ((maptable[observations_[j]]>=M) || (maptable[observations_[j]]==MAPTABLE_UNDEF))
	    {
	      fprintf(stderr,"wrong: %c -> %i\n",(char)observations_[j],(int)maptable[observations_[j]]) ;
	      fac=-1 ;
	    } ;
	  value= (value >> MAX_M) | (maptable[observations_[j]] << (MAX_M * (ORDER-1)));
	} ;
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

  for (i=ORDER-1; i<sequence_length; i++)	
    observations_[i-(ORDER-1)]=observations_[i];

  return fac*(sequence_length-(ORDER-1)) ;
}

//translate ACGT <-> 0123 in observations
void CObservation::translate_to_single_order(T_OBSERVATIONS* observations_, int sequence_length)
{
	T_OBSERVATIONS mask=(1 << MAX_M ) - 1;

	for (int i=0; i<sequence_length; i++)
		observations_[i]=maptable[mask & observations_[i]];
}
