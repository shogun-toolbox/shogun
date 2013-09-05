require 'nmatrix'
require 'modshogun'

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
    n = 0
    File.open(filename) do |file|
      file.each_line do |line|
        line.split(" ").each{ |n| matrix << n.to_f }
        n += 1
      end
    end
    m = matrix.length / n
    matrix = NMatrix.new([n,m], matrix, :float32)
    matrix = matrix.transpose()
  end

  def load_dna(filename)
    puts 'loading dna'
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
    vector = NMatrix.new([1, vector.length], vector, :int32)
  end
  extend self
end
