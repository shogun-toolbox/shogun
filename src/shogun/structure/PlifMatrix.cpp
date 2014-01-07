#include <structure/PlifMatrix.h>
#include <structure/Plif.h>
#include <structure/PlifArray.h>
#include <structure/PlifBase.h>

using namespace shogun;

CPlifMatrix::CPlifMatrix() : m_PEN(NULL), m_num_plifs(0), m_num_limits(0),
	m_num_states(0), m_feat_dim3(0), m_plif_matrix(NULL), m_state_signals(NULL)
{
}

CPlifMatrix::~CPlifMatrix()
{
	for (int32_t i=0; i<m_num_plifs; i++)
		delete m_PEN[i];
	SG_FREE(m_PEN);

	for (int32_t i=0; i<m_num_states*m_num_states; i++)
		delete m_plif_matrix[i];

	SG_FREE(m_plif_matrix);

	SG_FREE(m_state_signals);
}

void CPlifMatrix::create_plifs(int32_t num_plifs, int32_t num_limits)
{
	for (int32_t i=0; i<m_num_plifs; i++)
		delete m_PEN[i];
	SG_FREE(m_PEN);
	m_PEN=NULL;

	m_num_plifs=num_plifs;
	m_num_limits=num_limits;
	m_PEN = SG_MALLOC(CPlif*, num_plifs);
	for (int32_t i=0; i<num_plifs; i++)
		m_PEN[i]=new CPlif(num_limits) ;
}

void CPlifMatrix::set_plif_ids(SGVector<int32_t> plif_ids)
{
	if (plif_ids.vlen!=m_num_plifs)
		SG_ERROR("plif_ids size mismatch (num_ids=%d vs.num_plifs=%d)\n", plif_ids.vlen, m_num_plifs)

	m_ids.resize_array(m_num_plifs);
	m_ids.set_array(plif_ids.vector, plif_ids.vlen, true, true);

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		m_PEN[id]->set_id(id);
	}
}

void CPlifMatrix::set_plif_min_values(SGVector<float64_t> min_values)
{
	if (min_values.vlen!=m_num_plifs)
		SG_ERROR("plif_values size mismatch (num_values=%d vs.num_plifs=%d)\n", min_values.vlen, m_num_plifs)

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		m_PEN[id]->set_min_value(min_values.vector[i]);
	}
}

void CPlifMatrix::set_plif_max_values(SGVector<float64_t> max_values)
{
	if (max_values.vlen!=m_num_plifs)
		SG_ERROR("plif_values size mismatch (num_values=%d vs.num_plifs=%d)\n", max_values.vlen, m_num_plifs)

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		m_PEN[id]->set_max_value(max_values.vector[i]);
	}
}

void CPlifMatrix::set_plif_use_cache(SGVector<bool> use_cache)
{
	if (use_cache.vlen!=m_num_plifs)
		SG_ERROR("plif_values size mismatch (num_values=%d vs.num_plifs=%d)\n", use_cache.vlen, m_num_plifs)

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		m_PEN[id]->set_use_cache(use_cache.vector[i]);
	}
}

void CPlifMatrix::set_plif_use_svm(SGVector<int32_t> use_svm)
{
	if (use_svm.vlen!=m_num_plifs)
		SG_ERROR("plif_values size mismatch (num_values=%d vs.num_plifs=%d)\n", use_svm.vlen, m_num_plifs)

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		m_PEN[id]->set_use_svm(use_svm.vector[i]);
	}
}

void CPlifMatrix::set_plif_limits(SGMatrix<float64_t> limits)
{
	if (limits.num_rows!=m_num_plifs ||  limits.num_cols!=m_num_limits)
	{
		SG_ERROR("limits size mismatch expected (%d,%d) got (%d,%d)\n",
				m_num_plifs, m_num_limits, limits.num_rows, limits.num_cols);
	}

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		SGVector<float64_t> lim(m_num_limits);
		for (int32_t k=0; k<m_num_limits; k++)
			lim[k] = limits.matrix[i*m_num_limits+k];

		int32_t id=get_plif_id(i);
		m_PEN[id]->set_plif_limits(lim);
	}
}

void CPlifMatrix::set_plif_penalties(SGMatrix<float64_t> penalties)
{
	if (penalties.num_rows!=m_num_plifs ||  penalties.num_cols!=m_num_limits)
	{
		SG_ERROR("penalties size mismatch expected (%d,%d) got (%d,%d)\n",
				m_num_plifs, m_num_limits, penalties.num_rows, penalties.num_cols);
	}

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		SGVector<float64_t> pen(m_num_limits);

		for (int32_t k=0; k<m_num_limits; k++)
			pen[k] = penalties.matrix[i*m_num_limits+k];

		int32_t id=get_plif_id(i);
		m_PEN[id]->set_plif_penalty(pen);
	}
}

void CPlifMatrix::set_plif_names(SGString<char>* names, int32_t num_values, int32_t maxlen)
{
	if (num_values!=m_num_plifs)
		SG_ERROR("names size mismatch (num_values=%d vs.num_plifs=%d)\n", num_values, m_num_plifs)

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		char* name = CStringFeatures<char>::get_zero_terminated_string_copy(names[i]);
		m_PEN[id]->set_plif_name(name);
		SG_FREE(name);
	}
}

void CPlifMatrix::set_plif_transform_type(SGString<char>* transform_type, int32_t num_values, int32_t maxlen)
{
	if (num_values!=m_num_plifs)
		SG_ERROR("transform_type size mismatch (num_values=%d vs.num_plifs=%d)\n", num_values, m_num_plifs)

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		char* transform_str=CStringFeatures<char>::get_zero_terminated_string_copy(transform_type[i]);

		if (!m_PEN[id]->set_transform_type(transform_str))
		{
			SG_FREE(m_PEN);
			m_PEN=NULL;
			m_num_plifs=0;
			m_num_limits=0;
			SG_ERROR("transform type not recognized ('%s')\n", transform_str)
		}
		SG_FREE(transform_str);
	}
}


bool CPlifMatrix::compute_plif_matrix(SGNDArray<float64_t> penalties_array)
{
	CPlif** PEN = get_PEN();
	int32_t num_states = penalties_array.dims[0];
	int32_t num_plifs = get_num_plifs();

	for (int32_t i=0; i<m_num_states*m_num_states; i++)
		delete  m_plif_matrix[i];
	SG_FREE(m_plif_matrix);

	m_num_states = num_states;
	m_plif_matrix = SG_MALLOC(CPlifBase*, num_states*num_states);

	CDynamicArray<float64_t> penalties(penalties_array.array, num_states, num_states, penalties_array.dims[2], true, true) ;

	for (int32_t i=0; i<num_states; i++)
	{
		for (int32_t j=0; j<num_states; j++)
		{
			CPlifArray * plif_array = NULL;
			CPlif * plif = NULL ;
			for (int32_t k=0; k<penalties_array.dims[2]; k++)
			{
				if (penalties.element(i,j,k)==0)
					continue ;

				if (!plif_array)
				{
					plif_array = new CPlifArray() ;
					plif_array->clear() ;
				}

				int32_t id = (int32_t) penalties.element(i,j,k)-1 ;

				if ((id<0 || id>=num_plifs) && (id!=-1))
				{
					SG_ERROR("id out of range\n")
					CPlif::delete_penalty_struct(PEN, num_plifs) ;
					return false ;
				}
				plif = PEN[id] ;

				plif_array->add_plif(plif) ;
			}

			if (!plif_array)
			{
				m_plif_matrix[i+j*num_states] = NULL ;
			}
			else if (plif_array->get_num_plifs()==1)
			{
				SG_UNREF(plif_array);
				ASSERT(plif!=NULL)
				m_plif_matrix[i+j*num_states] = plif ;
			}
			else
				m_plif_matrix[i+j*num_states] = plif_array ;

		}
	}
	return true;
}

bool  CPlifMatrix::compute_signal_plifs(SGMatrix<int32_t> state_signals)
{
	int32_t Nplif = get_num_plifs();
	CPlif** PEN = get_PEN();

	SG_FREE(m_state_signals);
	m_feat_dim3 = state_signals.num_rows;

	CPlifBase **PEN_state_signal = SG_MALLOC(CPlifBase*, state_signals.num_rows*state_signals.num_cols);
	for (int32_t i=0; i<state_signals.num_cols*state_signals.num_rows; i++)
	{
		int32_t id = (int32_t) state_signals.matrix[i]-1 ;
		if ((id<0 || id>=Nplif) && (id!=-1))
		{
			SG_ERROR("id out of range\n")
			CPlif::delete_penalty_struct(PEN, Nplif) ;
			return false ;
		}
		if (id==-1)
			PEN_state_signal[i]=NULL ;
		else
			PEN_state_signal[i]=PEN[id] ;
	}
	m_state_signals=PEN_state_signal;
	return true;
}

void CPlifMatrix::set_plif_state_signal_matrix(
	int32_t *plif_id_matrix, int32_t m, int32_t max_num_signals)
{
	if (m!=m_num_plifs)
		SG_ERROR("plif_state_signal_matrix size does not match previous info %i!=%i\n", m, m_num_plifs)

	/*if (m_seq.get_dim3() != max_num_signals)
		SG_ERROR("size(plif_state_signal_matrix,2) does not match with size(m_seq,3): %i!=%i\nSorry, Soeren... interface changed\n", m_seq.get_dim3(), max_num_signals)

	CArray2<int32_t> id_matrix(plif_id_matrix, m_num_plifs, max_num_signals, false, false) ;
	m_PEN_state_signals.resize_array(m_num_plifs, max_num_signals) ;
	for (int32_t i=0; i<m_num_plifs; i++)
	{
		for (int32_t j=0; j<max_num_signals; j++)
		{
			if (id_matrix.element(i,j)>=0)
				m_PEN_state_signals.element(i,j)=m_plif_list[id_matrix.element(i,j)] ;
			else
				m_PEN_state_signals.element(i,j)=NULL ;
		}
	}*/
}
