/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#ifndef __CEREAL_VISITOR_H__
#define __CEREAL_VISITOR_H__

#include <shogun/lib/any.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/DataType.h>

namespace shogun
{
// forward declare the create factory in class_list.h
CSGObject* create(const char* sgserializable_name, EPrimitiveType generic);

template<class Archive>
class CerealWriterVisitor : public AnyVisitor
{
public:
	CerealWriterVisitor(Archive& ar) : AnyVisitor(), m_archive(ar)
	{
	}

	virtual void on(bool* v)
	{
		SG_SDEBUG("writing bool with value %d\n", *v)
		m_archive(*v);
	}
	virtual void on(int32_t* v)
	{
		SG_SDEBUG("writing int32_t with value %d\n", *v)
		m_archive(*v);
	}
	virtual void on(int64_t* v)
	{
		SG_SDEBUG("writing int64_t with value %d\n", *v)
		m_archive(*v);
	}
	virtual void on(float* v)
	{
		SG_SDEBUG("writing float with value %f\n", *v)
		m_archive(*v);
	}
	virtual void on(double* v)
	{
		SG_SDEBUG("writing double with value %f\n", *v)
		m_archive(*v);
	}
	virtual void on(CSGObject** v)
	{
		if (*v)
		{
			SG_SDEBUG("writing SGObject with of type\n")
			m_archive(**v);
		}
	}
	virtual void on(SGVector<int>* v)
	{
		SG_SDEBUG("writing SGVector<int>\n")
		m_archive(*v);
	}
	virtual void on(SGVector<float>* v)
	{
		SG_SDEBUG("writing SGVector<float>\n")
		m_archive(*v);
	}
	virtual void on(SGVector<double>* v)
	{
		SG_SDEBUG("writing SGVector<double>\n")
		m_archive(*v);
	}
	virtual void on(SGMatrix<int>* v)
	{
		SG_SDEBUG("writing SGMatrix<int>\n")
		m_archive(*v);
	}
	virtual void on(SGMatrix<float>* v)
	{
		SG_SDEBUG("writing SGMatrix<float>\n")
		m_archive(*v);
	}
	virtual void on(SGMatrix<double>* v)
	{
		SG_SDEBUG("writing SGMatrix<double>\n")
		m_archive(*v);
	}

private:
	Archive& m_archive;
};

template<class Archive>
class CerealReaderVisitor : public AnyVisitor
{
public:
	CerealReaderVisitor(Archive& ar) : AnyVisitor(), m_archive(ar)
	{
	}

	virtual void on(bool* v)
	{
		SG_SDEBUG("reading bool")
		*v = deserialize<bool>();
		SG_SDEBUG("%d\n", *v)
	}
	virtual void on(int32_t* v)
	{
		SG_SDEBUG("reading int32_t")
		*v = deserialize<int32_t>();
		SG_SDEBUG("%d\n", *v)
	}
	virtual void on(int64_t* v)
	{
		SG_SDEBUG("reading int64_t")
		*v = deserialize<int64_t>();
		SG_SDEBUG("%d\n", *v)
	}
	virtual void on(float* v)
	{
		SG_SDEBUG("reading float: ")
		*v = deserialize<float>();
		SG_SDEBUG("%f\n", *v)
	}
	virtual void on(double* v)
	{
		SG_SDEBUG("reading double: ")
		*v = deserialize<double>();
		SG_SDEBUG("%f\n", *v)
	}
	virtual void on(CSGObject** v)
	{
		SG_SDEBUG("reading SGObject: ")
		std::string object_name;
		EPrimitiveType primitive_type;
		m_archive(object_name, primitive_type);
		SG_SDEBUG("%s %d\n", object_name.c_str(), primitive_type)
		/* FIXME: what if storage is set,
		 * need to clear the storage before overwriting.
		if (*v == nullptr)
			SG_UNREF(*v);
		*/
		*v = create(object_name.c_str(), primitive_type);
		m_archive(**v);
	}
	virtual void on(SGVector<int>* v)
	{
		SG_SDEBUG("reading SGVector<int>\n")
		*v = deserialize<SGVector<int>>();
	}
	virtual void on(SGVector<float>* v)
	{
		SG_SDEBUG("reading SGVector<float>\n")
		*v = deserialize<SGVector<float>>();
	}
	virtual void on(SGVector<double>* v)
	{
		SG_SDEBUG("reading SGVector<double>\n")
		*v = deserialize<SGVector<double>>();
	}
	virtual void on(SGMatrix<int>* v)
	{
		SG_SDEBUG("reading SGMatrix<int>>\n")
		*v = deserialize<SGMatrix<int>>();
	}
	virtual void on(SGMatrix<float>* v)
	{
		SG_SDEBUG("reading SGMatrix<float>>\n")
		*v = deserialize<SGMatrix<float>>();
	}
	virtual void on(SGMatrix<double>* v)
	{
		SG_SDEBUG("reading SGMatrix<double>>\n")
		*v = deserialize<SGMatrix<double>>();
	}

private:
	template <typename T>
	T deserialize()
	{
		T value;
		m_archive(value);
		return value;
	}

private:
	Archive& m_archive;
};

}

#endif
