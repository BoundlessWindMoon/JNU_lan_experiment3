EXECUTABLE=conv2ddemo
BUILD_DIR := ./build
SRC_DIRS := ./src

SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' )
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

CXXFLAGS += -DHIP_ROCM -DNDEBUG -DUSE_DEFAULT_STDLIB   --offload-arch=gfx928 -g
# CXXFLAGS += -DHIP_ROCM -DNDEBUG -DUSE_DEFAULT_STDLIB -g
CC=$(HIP_PATH)/bin/hipcc
INCLUDES  += -I$(HIP_PATH)/include -I./include
LDFLAGS =

$(EXECUTABLE): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(EXECUTABLE)
      
	
$(BUILD_DIR)/%.cpp.o:%.cpp
	mkdir -p $(dir $@)
	$(CC) -c -w $< $(CXXFLAGS) $(INCLUDES) -o $@
	
.PHONY: clean job prof
LOG_DIR := ./log
PROF_DIR := ./prof
DUMP_DIR := ./assembly
PROF:= hipprof.sh
JOB := job.sh
DUMP := objdump.sh
TIMESTAMP := $(shell date '+%Y-%m-%d_%H-%M-%S')

job:
	mkdir -p $(LOG_DIR)
	sbatch -o $(LOG_DIR)/$(TIMESTAMP) $(JOB) 

prof:
	mkdir -p $(PROF_DIR)
	mkdir -p $(LOG_DIR)
	sbatch -o $(LOG_DIR)/$(TIMESTAMP) $(PROF)

dump:
	mkdir -p $(LOG_DIR)
	mkdir -p $(DUMP_DIR)
	sbatch -o $(LOG_DIR)/$(TIMESTAMP) $(DUMP)
	

clean:
	rm -rf $(BUILD_DIR) 
	rm ./*.txt

clean-all:
	rm -rf $(BUILD_DIR)
	rm -rf $(LOG_DIR)
	rm -rf $(PROF_DIR)
	rm -rf $(DUMP_DIR)


