EXECUTABLE=conv2ddemo
BUILD_DIR := ./build
SRC_DIRS := ./src

SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' )
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

# CXXFLAGS += -DHIP_ROCM -DNDEBUG -DUSE_DEFAULT_STDLIB   --amdgpu-target=gfx928 -g
CC=$(HIP_PATH)/bin/hipcc
CXXFLAGS += -DHIP_ROCM -DNDEBUG -DUSE_DEFAULT_STDLIB -g
INCLUDES  += -I$(HIP_PATH)/include -I./include
LDFLAGS =

$(BUILD_DIR)/$(EXECUTABLE): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(BUILD_DIR)/$(EXECUTABLE)
      
	
$(BUILD_DIR)/%.cpp.o:%.cpp
	mkdir -p $(dir $@)
	$(CC) -c -w $< $(CXXFLAGS) $(INCLUDES) -o $@
	
.PHONY: clean job prof
LOG_DIR := ./log
PROF_DIR := ./prof
PROF:= hipprof.sh
JOB := job.sh
TIMESTAMP := $(shell date '+%Y-%m-%d_%H-%M-%S')

job:
	mkdir -p $(LOG_DIR)
	sbatch -o $(LOG_DIR)/$(TIMESTAMP) $(JOB) 

prof:
	mkdir -p $(PROF_DIR)
	mkdir -p $(LOG_DIR)
	sbatch -o $(LOG_DIR)/$(TIMESTAMP) $(PROF)

clean:
	rm -rf $(BUILD_DIR)

clean-all:
	rm -rf $(BUILD_DIR)
	rm -rf $(LOG_DIR)
	rm -rf $(PROF_DIR)

