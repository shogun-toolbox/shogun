#include "structure/PlifMatrix.h"
#include "structure/Plif.h"
#include "structure/PlifArray.h"
#include "structure/PlifBase.h"
#include "lib/Array3.h"

CPlifMatrix::CPlifMatrix() : m_PEN(NULL), m_num_plifs(0), m_num_limits(0),
	m_plif_matrix(NULL), m_state_signals(NULL)
{
}

CPlifMatrix::~CPlifMatrix()
{
}

bool CPlifMatrix::set_plif_struct(
	int32_t N, int32_t M, float64_t* all_limits, float64_t* all_penalties,
	int32_t* ids, T_STRING<char>* names, float64_t* min_values,
	float64_t* max_values, bool* all_use_cache, int32_t* all_use_svm,
	T_STRING<char>* all_transform)
{
	// cleanup 
	//SG_PRINT("set_plif_struct, N:%i\n",N);
	for (int32_t i=0; i<m_num_plifs; i++)	
		delete m_PEN[i];
	delete[] m_PEN;
	m_PEN=NULL;

	// init values
	m_num_plifs=N;
	m_num_limits=M;
	m_PEN = new CPlif*[N] ;
	for (int32_t i=0; i<N; i++)	
		m_PEN[i]=new CPlif(M) ;

	for (int32_t i=0; i<N; i++)
	{
		float64_t* limits = new float64_t[M];
		float64_t* penalties = new float64_t[M];
		for (int32_t k=0; k<M; k++)
		{
			limits[k] = all_limits[i*M+k];
			penalties[k] = all_penalties[i*M+k];
		}
		int32_t id = ids[i];
		if (id>=N)
			SG_ERROR("plif id (%i)  exceeds array length (%i)\n",id,N);
		m_PEN[id]->set_id(id);

		m_PEN[id]->set_plif_name(get_zero_terminated_string_copy(names[i]));
		m_PEN[id]->set_min_value(min_values[i]);
		m_PEN[id]->set_max_value(max_values[i]);
		m_PEN[id]->set_use_cache(all_use_cache[i]);
		m_PEN[id]->set_use_svm(all_use_svm[i]);
		m_PEN[id]->set_plif_limits(limits, M);
		m_PEN[id]->set_plif_penalty(penalties, M);
		//m_PEN[id]->set_do_calc(all_do_calc[i]); //JONAS FIX
		char* transform_str=get_zero_terminated_string_copy(all_transform[i]);
		if (!m_PEN[id]->set_transform_type(transform_str))
		{
			SG_ERROR( "transform type not recognized ('%s')\n", transform_str) ;
			delete[] m_PEN;
			m_PEN=NULL;
			m_num_plifs=0;
			m_num_limits=0;
			return false;
		}
	}

	return true;
}


bool CPlifMatrix::compute_plif_matrix(
	float64_t* penalties_array, int32_t* Dim, int32_t numDims)
{
	CPlif** PEN = get_PEN();
	int32_t num_states = Dim[0];
	int32_t num_plifs = get_num_plifs();

	delete[] m_plif_matrix ;
	m_plif_matrix = new CPlifBase*[num_states*num_states] ;

	CArray3<float64_t> penalties(penalties_array, num_states, num_states, Dim[2], true, true) ;

	for (int32_t i=0; i<num_states; i++)
	{
		for (int32_t j=0; j<num_states; j++)
		{
			CPlifArray * plif_array = new CPlifArray() ;
			CPlif * plif = NULL ;
			plif_array->clear() ;
			for (int32_t k=0; k<Dim[2]; k++)
			{
				if (penalties.element(i,j,k)==0)
					continue ;
				int32_t id = (int32_t) penalties.element(i,j,k)-1 ;

				if ((id<0 || id>=num_plifs) && (id!=-1))
				{
					SG_ERROR( "id out of range\n") ;
					CPlif::delete_penalty_struct(PEN, num_plifs) ;
					return false ;
				}
				plif = PEN[id] ;

				plif_array->add_plif(plif) ;
			}

			if (plif_array->get_num_plifs()==0)
			{
				SG_UNREF(plif_array);
				m_plif_matrix[i+j*num_states] = NULL ;
			}
			else if (plif_array->get_num_plifs()==1)
			{
				SG_UNREF(plif_array);
				ASSERT(plif!=NULL) ;
				m_plif_matrix[i+j*num_states] = plif ;
			}
			else
				m_plif_matrix[i+j*num_states] = plif_array ;

		}
	}
	return true;
}

bool  CPlifMatrix::compute_signal_plifs(
	int32_t* state_signals, int32_t feat_dim3, int32_t num_states)
{
	int32_t Nplif = get_num_plifs();
	CPlif** PEN = get_PEN();

	CPlifBase **PEN_state_signal = new CPlifBase*[feat_dim3*num_states] ;
	for (int32_t i=0; i<num_states*feat_dim3; i++)
	{
		int32_t id = (int32_t) state_signals[i]-1 ;
		if ((id<0 || id>=Nplif) && (id!=-1))
		{
			SG_ERROR( "id out of range\n") ;
			CPlif::delete_penalty_struct(PEN, Nplif) ;
			return false ;
		}
		if (id==-1)
			PEN_state_signal[i]=NULL ;
		else
			PEN_state_signal[i]=PEN[id] ;
	}
	set_state_signals(PEN_state_signal);
	return true;
}
