CXXFLAGS=-O9 -g $(INCLUDES)
LINKFLAGS=-lshogun
TARGETS = vw

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS): $(foreach t, $(TARGET), $t.cpp)

%: %.cpp
	gcc $(CXXFLAGS) $< $(LINKFLAGS) -lshogun -o $@
