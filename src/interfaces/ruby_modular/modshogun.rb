require 'narray'
require 'modshogun.so'

def LoadMatrix(filename, type = :numbers)
  case type
  when :numbers
    x = LoadMatrix.load_numbers(filename)
    pp x
  when :dna
    LoadMatrix.load_dna(filename)
  when :cubes
    LoadMatrix.load_cubes(filename)
  when :labels
    LoadMatrix.load_labels(filename)
  end
end

module LoadMatrix
  def load_numbers(filename)
    matrix = []
    File.open(filename) do |file|
      file.each_line do |line|
        ary = []
        line.split(" ").each{ |n| ary << n.to_f }
        matrix << ary
      end
    end
    #pp matrix
    matrix = NArray.to_na(matrix)
    #pp matrix
    matrix = matrix.transpose(1,0)
    #pp matrix
  end

  def load_dna(filename)
    matrix=[]
    File.open(filename) do |file|
      file.each_line do |line|
        line.split("\n").each{|n| matrix << n }
      end
    end
    matrix
  end


  def load_cubes(filename)
    matrix=[]
    File.open(filename) do |file|
      file.each_line do |line|
        line.split("\n").each{|n| matrix << n }
      end
    end
    matrix
  end


  def load_labels(filename)
    vector = []
    File.open(filename) do |file|
      file.each_line do |line|
        vector << line.to_f
      end
    end
    vector = NArray.to_na(vector)
    # loading vectors is not yet implemented,
    # use a 1D array instead (for now)
    #vector = vector.reshape(1, vector.total)
  end
 extend self
end
