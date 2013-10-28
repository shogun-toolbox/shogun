require 'os'
require 'modshogun'
require 'load'

parameter_list={{5,1,10, 2.0, 10}, {10,0.3,2, 1.0, 0.1}}

function check_status(status)

	assert(status == true)
	-- if  status:
  --	print "OK reading/writing .h5\n"
	--else:
	--	print "ERROR reading/writing .h5\n"
end

function concatenate(...)
	local result = ...
	for _,t in ipairs{select(2, ...)} do
		for row,rowdata in ipairs(t) do
			for col,coldata in ipairs(rowdata) do
				table.insert(result[row], coldata)
			end
		end
	end
	return result
end

function rand_matrix(rows, cols, dist)
  local matrix = {}
	for i = 1, rows do
		matrix[i] = {}
		for j = 1, cols do
			matrix[i][j] = math.random() + dist
		end
	end
	return matrix
end

function generate_lab(num)
	lab={}
	for i=1,num do
		lab[i]=0
	end
	for i=num+1,2*num do
		lab[i]=1
	end
	for i=2*num+1,3*num do
		lab[i]=2
	end
	for i=3*num+1,4*num do
		lab[i]=3
	end
	return lab
end

function serialization_complex_example(num, dist, dim, C, width)

	math.randomseed(17)

	data=concatenate(rand_matrix(dim, num, 0), rand_matrix(dim, num, dist), rand_matrix(dim, num, 2 * dist), rand_matrix(dim, num, 3 * dist))

	lab=generate_lab(num)

	feats=modshogun.RealFeatures(data)
	kernel=modshogun.GaussianKernel(feats, feats, width)

	labels=modshogun.MulticlassLabels(lab)

	svm = modshogun.GMNPSVM(C, kernel, labels)

	feats:add_preprocessor(modshogun.NormOne())
	feats:add_preprocessor(modshogun.LogPlusOne())
	feats:set_preprocessed(1)
	svm:train(feats)

	fstream = modshogun.SerializableHdf5File("blaah.h5", "w")
	status = svm:save_serializable(fstream)
	check_status(status)

	fstream = modshogun.SerializableAsciiFile("blaah.asc", "w")
	status = svm:save_serializable(fstream)
	check_status(status)

	-- fstream = modshogun.SerializableJsonFile("blaah.json", "w")
	-- status = svm:save_serializable(fstream)
	-- check_status(status)

	fstream = modshogun.SerializableXmlFile("blaah.xml", "w")
	status = svm:save_serializable(fstream)
	check_status(status)


	fstream = modshogun.SerializableHdf5File("blaah.h5", "r")
	new_svm=modshogun.GMNPSVM()
	status = new_svm:load_serializable(fstream)
	check_status(status)
	new_svm:train()

	fstream = modshogun.SerializableAsciiFile("blaah.asc", "r")
	new_svm=modshogun.GMNPSVM()
	status = new_svm:load_serializable(fstream)
	check_status(status)
	new_svm:train()

	-- fstream = modshogun.SerializableJsonFile("blaah.json", "r")
	-- new_svm=modshogun.GMNPSVM()
	-- status = new_svm:load_serializable(fstream)
	-- check_status(status)
	-- new_svm:train()

	fstream = modshogun.SerializableXmlFile("blaah.xml", "r")
	new_svm=modshogun.GMNPSVM()
	status = new_svm:load_serializable(fstream)
	check_status(status)
	new_svm:train()

	os.remove("blaah.h5")
	os.remove("blaah.asc")
	-- os.remove("blaah.json")
	os.remove("blaah.xml")

	return svm,new_svm
end

if debug.getinfo(3) == nill then
	print 'Serialization SVMLight'
	serialization_complex_example(unpack(parameter_list[1]))
end
