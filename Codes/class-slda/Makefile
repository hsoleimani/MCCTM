CC = g++
LDFLAGS = -lgsl -lm -lgslcblas

GSL_INCLUDE = /usr/global/gsl/1.16/include/
GSL_LIB = /usr/global/gsl/1.16/lib

LSOURCE = main.cpp corpus.cpp slda.cpp utils.cpp opt.cpp lda-alpha.cpp
LHEADER = corpus.h slda.h utils.h opt.h settings.h lda-alpha.h

all: $(LSOURCE) $(HEADER)
	  $(CC) -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) $(LDFLAGS) -o SLDA 

clean:
	-rm -f *.o slda
