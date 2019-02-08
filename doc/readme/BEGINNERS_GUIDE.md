Beginner's Guide
---
This is a beginner's guide to prople getting started to open source developement or are in phase of developement.
for those of you reading this , are quite seasoned to open-sourse and the wraths of open-source might want to jump ahed to the [ detailed guide](https://github.com/shogun-toolbox/shogun/blob/develop/README.md) , or for those who want to test shogun might want to test shogun and its ready-to-install packages might want to check [ready-to-install](https://github.com/shogun-toolbox/shogun/blob/develop/doc/readme/INSTALL.md) .

Starting off with your contribution would be installation / setting up your device to run shogun as a library for any of the following environments supported :
 - C++ (default installation)
 - Python
 - Octave
 - R
 - Java
 - Ruby
 - Lua
 - C-Sharp
 
 ### Installation 
For developement and contribution of this project its recommended that you go through these steps for installation :
-	***Step 1* : Solving Dependencies**
	To , make things work fast and giving you more space 			  to pivot in further maturity in this project its 		  recommended that you install all of the following  installed for **dependencies** :
	- Python
		    `python-dev python-numpy`
	- Octave
			`octave liboctave-dev`
	- R
	    `r-base-core`
	-   Java
	       `oracle-java8-installer`
	- Ruby
		  `ruby ruby-dev`, and  `narray`
	- Lua
			`lua5.1 liblua5.1-0-dev`
	- C-Sharp
		   `mono-devel mono-gmcs cli-common-dev`
----
Make sure that you have cmake or  installed or ccmake installed by :
	`cmake --version`
	else install it :
	`sudo apt-get install cmake`
		
---
- ***Step 2* : Cloning in the source files for shogun** 
	Assuming you have `git` installed :
	-	Clone in shogun source code using :
		``
	git clone https://github.com/shogun-toolbox/shogun.git ``
		``
	cd shogun
	``
		``git submodule update --init``
---
- ***Step 3 :* Building and compiling** 
	- Create the build directory and change pwd to `build` :
		`mkdir build ; cd build`
Configure cmake, from the build directory, passing the Shogun source root as argument. It is recommended to use any of CMake GUIs (e.g. replace `cmake ..` with `ccmake ..`), in particular if you feel unsure about possible parameters and configurations. Note that all cmake options read as `-DOPTION=VALUE`.
	If you are new using cmake :
		- if you want to install C++ only (enabled by default) :
			`cmake BUILD_META_EXAMPLE=ON ..`
		- if you want to get examples containing python code :
			`cmake BUILD_META_EXAMPLES=ON -DINTERFACE_PYTHON=ON ..`
		- else if for any other interfaces in the set (`R,Ruby,Octave,Java,Lua,C-sharp`) 
			considering replacing the `<interface_wanted>` with the required / desired interface :
			`cmake BUILD_META_EXAMPLES=ON -DINTERFACE_<interface_wanted>=ON ..`
	- Compiling 
		``
make
``

		Install (prepend  `sudo`  if installing system wide), and your are done !
`make install`
Sometimes you might need to clean up your build (e.g. in case of some major changes). First, try
`make clean`
If that does not help, try removing the build directory and starting from scratch afterwards
`rm -rf build`

If you prefer to not run the  `sudo make install`  command system wide, you can either install Shogun to a custom location (`-DCMAKE_INSTALL_PREFIX=/custom/path`, defaults to  `/usr/local`), or even skip  `make install`  at all. In both cases, it is necessary to set a number of system libraries for using Shogun, see  [INTERFACES.md](https://github.com/shogun-toolbox/shogun/blob/develop/doc/readme/INTERFACES.md).

---
- ***Step 4* :  Test Examples** 
	After setting up , its for the best that you test the setup running a few examples locally located `/shogun/build/examples/meta/<whichever_interface_you_preffer>`
	
	For running the examples you must also have the interfaces setup as explained in the following block :
	## Setting up Interfaces

	We now describe how to run code that uses Shogun in all interfaces. For language specific defails how to import and use Shogun in all interfaces, see the examples on our website.

	### Native C++

	Make sure you read up on how to compile C/C++ code. Compilation requires the Shogun headers path, i.e. the 	path where for example  `shogun/base/init.h`  is located. This is either in  `path/to/src/shogun/`  or in  	`path/to/shogun-install/include/shogun/`  and is specified via the  `-I`  flag. Linking requires the  `-lshogun`  flag, which either needs the  `LD_LIBRARY_PATH`  or  `DYLD_LIBRARY_PATH`  set up as described above, or preferably passed via the  `-L`  flag. Compiling and linking code with  `gcc`  works as 
	`
gcc path/to/native_example.cpp -o native_example -I/path/to/headers -lshogun -L/path/to/libshogun.* `

	Running it:
	` ./native_example`

	### Python
	This needs  `shogun.py`  to be visible, which is either in  `path/to/build/src/interfaces/python/`  or in something similar to  `path/to/shogun-install/lib/python2.7/dist-packages/`
	would become as :
	`export PYTHONPATH="path/to/modshogun.py:$PYTHONPATH"`
	Running an example:
`python path/to/python_example.py`
