exe=kcore.bin
#exe=pr.bin

ldflags	= -I. -I../ -I../../lib/
#ldflags	+= -I../../test/pagerank
ldflags	+= -I../../test/kcore

commflags= -Wall 
cc= "$(shell which g++)"

ifeq ($(monitor), 1)
	cuflags+= -DENABLE_MONITORING
endif

ifeq ($(check), 1)
	cuflags+= -DENABLE_CHECKING
endif

ifeq ($(debug), 1)
	cuflags+= -G -g -O0
	cflags += -g -O0
else
	cflags += -O3
	cuflags+= -O3
endif


#cuflags+= -ccbin=g++ -Xcompiler -fopenmp

objs	= $(patsubst %.cpp,%.o,$(wildcard *.cpp))

deps	= $(wildcard ./*.hpp) \
		  $(wildcard ./*.h) \
		  $(wildcard ../*.hpp) \
		  Makefile

%.o:%.cpp $(deps)
	$(cc) -c $(cuflags) $(ldflags) $< -o $@

$(exe):$(objs)
	$(cc) $(objs) $(cuflags) $(ldflags) -o $(exe)

test:$(exe)
	./$(exe) /lustre/groups/huanglab/discovery_dataset/livejournal/livejournal_beg_pos.bin /lustre/groups/huanglab/discovery_dataset/livejournal/livejournal_csr.bin /lustre/groups/huanglab/discovery_dataset/livejournal/livejournal_weight.bin 1

clean:
	rm -rf *.o generator/*.o ${exe}

