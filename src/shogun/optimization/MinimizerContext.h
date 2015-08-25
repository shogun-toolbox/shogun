/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 */

#ifndef MINIMIZERCONTEXT_H
#define MINIMIZERCONTEXT_H
#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/StringMap.h>

namespace shogun
{
class CMinimizerContext: public CSGObject
{
public:
	/*  Constructor */
	CMinimizerContext()
		:CSGObject()
	{
		init();
	}

	/*  Destructor */
	virtual ~CMinimizerContext()
	{
		SG_UNREF(m_int32_map);
		SG_UNREF(m_float64_map);
		SG_UNREF(m_sgvector_float64_map);
	}

	/** Returns the name of the inference method
	 *
	 * @return name MinimizerContext
	 */
	virtual const char* get_name() const {return "MinimizerContext";}

	virtual void save_data(const std::string& key, SGVector<float64_t> value)
	{
		REQUIRE(!m_sgvector_float64_map->contains(key),
			"Failed to save data due to duplicate key:%s\n", key.c_str());
		m_sgvector_float64_map->add(key, value);
		REQUIRE(m_sgvector_float64_map->contains(key),
			"Failed to save data for key:%s\n", key.c_str());
	}

	virtual void save_data(const std::string& key, float64_t value)
	{
		REQUIRE(!m_float64_map->contains(key),
			"Failed to save data due to duplicate key:%s\n", key.c_str());
		m_float64_map->add(key, value);
		REQUIRE(m_float64_map->contains(key),
			"Failed to save data for key:%s\n", key.c_str());
	}

	virtual void save_data(const std::string& key, int32_t value)
	{
		REQUIRE(!m_int32_map->contains(key),
			"Failed to save data due to duplicate key:%s\n", key.c_str());
		m_int32_map->add(key, value);
		REQUIRE(m_int32_map->contains(key),
			"Failed to save data for key:%s\n", key.c_str());
	}

	virtual SGVector<float64_t> get_SGVector_float64(const std::string& key)
	{
		REQUIRE(m_sgvector_float64_map->contains(key),
			"Failed to load data because key:%s does not exist\n", key.c_str());
		return m_sgvector_float64_map->get_element(key);
	}

	virtual float64_t get_float64(const std::string& key)
	{
		REQUIRE(m_float64_map->contains(key),
			"Failed to load data because key:%s does not exist\n", key.c_str());
		return m_float64_map->get_element(key);
	}

	virtual int32_t get_data_int32(const std::string& key)
	{
		REQUIRE(m_int32_map->contains(key),
			"Failed to load data because key:%s does not exist\n", key.c_str());
		return m_int32_map->get_element(key);
	}

protected:
	CStringMap<int32_t> *m_int32_map;
	CStringMap<float64_t> *m_float64_map;
	CStringMap< SGVector<float64_t> > *m_sgvector_float64_map;
private:
	/*  Init */
	void init()
	{
		m_float64_map=new CStringMap<float64_t>();
		m_int32_map=new CStringMap<int32_t>();
		m_sgvector_float64_map=new CStringMap< SGVector<float64_t> >();
		SG_REF(m_sgvector_float64_map);
		SG_REF(m_int32_map);
		SG_REF(m_float64_map);
		//TODO: uncomment the following lines when CMap (including CStringMap) supports load_serializable() and save_serializable()
		//SG_ADD((CSGObject **)&m_int32_map, "int32_map", "int32_map", MS_NOT_AVAILABLE);
		//SG_ADD((CSGObject **)&m_float64_map, "float64_map", "float64_map", MS_NOT_AVAILABLE);
		//SG_ADD((CSGObject **)&m_sgvector_float64_map, "sgvector_float64_map", "sgvector_float64_map", MS_NOT_AVAILABLE);
	}
	
};
}

#endif
