EXECUTABLE=conv2dfp16demo
BUILD_DIR := ./build
SRC_DIRS := src

SRCS := $(shell find $(SRC_DIRS) -name '*.cu' )
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
CXXFLAGS += -O3 -g -arch=sm_89

ifeq ($(TEST),y)
	 CXXFLAGS += -DTEST=y
endif

# CXXFLAGS += -DHIP_ROCM -DNDEBUG -DUSE_DEFAULT_STDLIB -g
CC=nvcc
INCLUDES += -I./include
LDFLAGS =

$(EXECUTABLE): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(EXECUTABLE)
      
$(BUILD_DIR)/%.cu.o:%.cu
	mkdir -p $(dir $@)
	$(CC) -c -w $< $(CXXFLAGS) $(INCLUDES) -o $@
	
.PHONY: clean test prof clean-all 
LOG_DIR := ./log
TEST := test.sh
COMMIT := commit.sh
TIMESTAMP := $(shell date '+%Y-%m-%d_%H-%M-%S')

commit:
	mkdir -p $(LOG_DIR)
	sh $(COMMIT) -o $(LOG_DIR)/$(TIMESTAMP) 

test:
	mkdir -p $(LOG_DIR)
	sh $(TEST) -o $(LOG_DIR)/$(TIMESTAMP) 	

clean:
	rm -rf $(BUILD_DIR) 
	rm ./$(EXECUTABLE)

clean-all:
	rm -rf $(BUILD_DIR)
	rm ./$(EXECUTABLE)
	rm -rf $(LOG_DIR)
	rm -rf $(PROF_DIR)

