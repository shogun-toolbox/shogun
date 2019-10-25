
#include <stdio.h>
#include <string.h>

#include <shogun/mathematics/Math.h>
#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/structure/IntronList.h>

using namespace shogun;

IntronList::IntronList()
:SGObject()
{
	m_length = 0;
	m_all_pos = NULL;
	m_intron_list = NULL;
	m_quality_list = NULL;
}
IntronList::~IntronList()
{
	for (int i=0; i<m_length; i++)
	{
		SG_FREE(m_intron_list[i]);
		SG_FREE(m_quality_list[i]);
	}
	SG_FREE(m_intron_list);
	SG_FREE(m_quality_list);
	SG_FREE(m_all_pos);
}
void IntronList::init_list(int32_t* all_pos, int32_t len)
{
	m_length = len;
	m_all_pos = SG_MALLOC(int32_t, len);
	sg_memcpy(m_all_pos, all_pos, len*sizeof(int32_t));
	m_intron_list = SG_MALLOC(int32_t*, len);
	m_quality_list = SG_MALLOC(int32_t*, len);

	//initialize all elements with an array of length one
	int32_t* one;
	for (int i=0;i<m_length;i++)
	{
		one = SG_MALLOC(int32_t, 1);//use malloc here because mem can be increased efficiently with realloc later
		m_intron_list[i] = one;
		m_intron_list[i][0] = 1;
		one = SG_MALLOC(int32_t, 1);
		m_quality_list[i] = one;
		m_quality_list[i][0] = 1;
	}
}
void IntronList::read_introns(int32_t* start_pos, int32_t* end_pos, int32_t* quality, int32_t len)
{
	int k=0;
	for(int i=0;i<m_length;i++)//iterate over candidate positions
	{
		while (k<len)
		{
			//io::print("i:{}, m_all_pos[i]:{}, k:{}, end_pos[k]: {}\n", i, m_all_pos[i], k, end_pos[k]);
			if (k>0)
				if (end_pos[k]<end_pos[k-1])
					error("end pos array is not sorted: end_pos[k-1]:{} end_pos[k]:{}", end_pos[k-1], end_pos[k]);
			if (end_pos[k]>=m_all_pos[i])
				break;
			else
				k++;

		}
		while (k<len && end_pos[k]==m_all_pos[i])
		{
			//io::print("\tk:{}, end_pos[k]: {}, start_pos[k]:{}\n", k, end_pos[k], start_pos[k]);
			ASSERT(start_pos[k]<end_pos[k])
			ASSERT(end_pos[k]<=m_all_pos[m_length-1])
			// intron list
			//------------
			int32_t from_list_len = m_intron_list[i][0];
			int32_t* new_list = SG_REALLOC(int32_t, m_intron_list[i], from_list_len, (from_list_len+1));
			if (new_list == NULL)
				error("IntronList: Out of mem 4");
			new_list[from_list_len]= start_pos[k];
			new_list[0]++;
			m_intron_list[i] = new_list;
			// quality list
			//--------------
			int32_t q_list_len = m_quality_list[i][0];
			//io::print("\t q_list_len:{}, from_list_len:{} \n",q_list_len, from_list_len);
			ASSERT(q_list_len==from_list_len)
			new_list = SG_REALLOC(int32_t, m_quality_list[i], q_list_len, (q_list_len+1));
			if (new_list == NULL)
				error("IntronList: Out of mem 5");
			new_list[q_list_len]= quality[k];
			new_list[0]++;
			m_quality_list[i] = new_list;

			k++;
		}
	}
}
/**
 * from_pos and to_pos are indices in the all_pos list
 * not positions in the DNA sequence
 * */
void IntronList::get_intron_support(int32_t* values, int32_t from_pos, int32_t to_pos)
{
	if (from_pos>=m_length)
		error("from_pos ({}) is not < m_length ({})",to_pos, m_length);
	if (to_pos>=m_length)
		error("to_pos ({}) is not < m_length ({})",to_pos, m_length);
	int32_t* from_list = m_intron_list[to_pos];
	int32_t* q_list = m_quality_list[to_pos];

	//io::print("from_list[0]: {}\n", from_list[0]);

	int32_t coverage = 0;
	int32_t quality = 0;
	for (int i=1;i<from_list[0]; i++)
	{
		//io::print("from_list[{}]: {}, m_all_pos[from_pos]:{}\n", i,  from_list[i], m_all_pos[from_pos]);
		if (from_list[i]==m_all_pos[from_pos])
		{
			//io::print("found intron: {}->{}\n", from_pos, to_pos );
			coverage = coverage+1;
			quality = Math::max(quality, q_list[i]);
		}
	}
	values[0] = coverage;
	values[1] = quality;
}
