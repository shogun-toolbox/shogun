CXXFLAGS=-O9 -g $(INCLUDES)
LINKFLAGS=-lshogun
TARGETS = sailfish

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS): $(foreach t, $(TARGET), $t.cpp)

%: %.cpp
	gcc $(CXXFLAGS) $< $(LINKFLAGS) -lshogun -o $@
