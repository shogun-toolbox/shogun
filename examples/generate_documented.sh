#!/usr/bin/env bash

static_dirs="cmdline_static elwms_static matlab_and_octave python_static r_static"
modular_dirs="octave_modular python_modular r_modular lua_modular csharp_modular ruby_modular java_modular"
lib_dirs=libshogun
rm -f missing.log

document_interface()
{
	iftype=$1

	if [ "$iftype" == "static" ]
	then
		if_dirs="${static_dirs}"
		if_descr_dir="static"

	elif [ "$iftype" == "modular" ]
	then
		if_dirs="${modular_dirs}"
		if_descr_dir="modular"

	elif [ "$iftype" == "lib" ]
	then
		if_dirs=""
		if_descr_dir="modular"
		rm -rf documented/${lib_dirs}
		mkdir documented/${lib_dirs}
		cp undocumented/${lib_dirs}/* documented/${lib_dirs}
	fi

	for d in ${if_dirs}
	do

		files=`ls -1 undocumented/$d | grep -v check.sh`
		rm -rf documented/$d
		mkdir documented/$d


		for f in $files
		do
			doc=`echo $f | cut -f 1 -d '.' | sed 's/_modular$//'`.txt
			#echo undocumented/$d/$f
			if test -f undocumented/$d/$f
			then
				if test -f descriptions/${if_descr_dir}/$doc
				then
					suffix=`echo $f | cut -f 2 -d '.'`

					case "$suffix" in
						(py)
							prefix="#"
							;;
						(sg)
							prefix="%"
							;;
						(m)
							prefix="%"
							;;
						(R)
							prefix="#"
							;;
						(rb)
							prefix="#"
							;;
						(lua)
							prefix="-- "
							;;
						(java)
							prefix="\/\/"
							;;
						(cs)
							prefix="\/\/"
							;;
					esac
					( cat descriptions/${if_descr_dir}/$doc | sed "s/^/$prefix /" ; \
						echo; \
						cat undocumented/$d/$f ) >documented/$d/$f
				else
					echo "documentation for \"$f\" (file \"descriptions/${if_descr_dir}/$doc\") missing" >>missing.log
					cat undocumented/$d/$f >documented/$d/$f
				fi
			fi
		done

		for vanilla in graphical tools
		do
			test -d undocumented/$d/$vanilla && \
				( mkdir documented/$d/$vanilla &&  \
				cp undocumented/$d/$vanilla/* documented/$d/$vanilla/ )
		done
	done
}

document_interface "static"
document_interface "modular"
document_interface "lib"
exit 0
