require 'modshogun'
require 'load'

strings = {'hey','guys','i','am','a','string'}
parameter_list={{strings}}

function features_string_char_modular(strings)
	--for k, v in pairs(strings) do print(v) end
	--FIXME
	--f=modshogun.StringCharFeatures(strings, modshogun.RAWBYTE)

	--print("max string length " ..f:get_max_vector_length())
	--print("number of strings " .. f:get_num_vectors())
	--print ("length of first string" ..f:get_vector_length(0))
	--print ("strings" .. f:get_features())

	--FIXME
	--f:set_feature_vector({"t","e","s","t"}, 0)

	--return f:get_features(), f
end

if debug.getinfo(3) == nill then
	print 'StringCharFeatures'
	features_string_char_modular(unpack(parameter_list[1]))
end
