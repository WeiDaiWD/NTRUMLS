#################################################
#################################################
# Compilers
CC	= gcc
CX	= g++
CU	= /usr/local/cuda/bin/nvcc
# Flags
FMACH	=  -m64
FOPTM	= -O2
FC++11	= -std=c++11
FOMP	= -fopenmp
FLAGS	= $(FOPTM) $(FMACH)
# CUDA Flags
ARCH	= -arch=compute_30
CODE	= -code=sm_30
PTX		= -code=compute_30
# Libraries
LCUDART	= --cudart static
LMATH	= -lm
LNTL	= -lntl
LGMP	= -lgmp
LOMP	= -lgomp
LINKERS	= $(MATH) $(LCUDART)
# Commands
COMPCC	= $(CC) $(FLAGS)
COMPCX	= $(CX) $(FLAGS)
COMPCU	= $(CU) $(FLAGS) -maxrregcount 32
LINKCC	= $(CC) $(FLAGS) -link -w
LINKCX	= $(CX) $(FLAGS) -link -w
LINKCU	= $(CU) $(FLAGS) $(ARCH) $(CODE) $(PTX) -link -w

#################################################
#################################################
# Source Files
SRCCC	=	src/bench.c \
			src/convert.c \
			src/crypto_hash_sha512.c \
			src/crypto_stream.c \
			src/fastrandombytes.c \
			src/pack.c \
			src/params.c \
			src/pol.c \
			src/pqntrusign.c \
			src/randombytes.c \
			src/shred.c

SRCCX	= 

SRCCU	= src/cuda_pqntrusign.cu

SRCS	= $(SRCCC) $(SRCCX) $(SRCCU)
# Dependency Files
DEPCC	= $(SRCCC:.c=.d)
DEPCX	= $(SRCCX:.cpp=.d)
DEPCU	= $(SRCCU:.cu=.d)
DEPS	= $(DEPCC) $(DEPCX) $(DEPCU)	
# Object Files
OBJCC	= $(SRCCC:.c=.o)
OBJCX	= $(SRCCX:.cpp=.o)
OBJCU	= $(SRCCU:.cu=.o)
OBJS	= $(OBJCC) $(OBJCX) $(OBJCU)
# Library Files
LIBS	= 
# Executable Targets
TARS	= bin/bench

#################################################
#################################################
all: $(TARS)

bin/bench: $(OBJS) $(LIBS)
	-mkdir -p bin
	$(LINKCU) -o $@ $(OBJS) $(LIBS) $(LINKERS)
#### Compile ####
$(OBJCC): $(@:%.o=%.c)
	$(COMPCC) -M $(@:%.o=%.c) -o $(@:%.o=%.d)
	$(COMPCC) -c $(@:%.o=%.c) -o $@
$(OBJCX): $(@:%.o=%.cpp)
	$(COMPCX) -M $(@:%.o=%.cpp) -o $(@:%.o=%.d)
	$(COMPCX) -c $(@:%.o=%.cpp) -o $@
$(OBJCU): $(@:%.o=%.cu)
	$(COMPCU) -M $(@:%.o=%.cu) -o $(@:%.o=%.d)
	$(COMPCU) -c $(@:%.o=%.cu) -o $@
#### Others ####
clean:
	rm $(DEPS) $(OBJS) $(TARS) -r bin

