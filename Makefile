PROJECT := tensor_scope
CONFIGS := Makefile.config

include $(CONFIGS)

OBJ_DIR := obj
SRC_DIR := src
INC_DIR := include
LIB_DIR := lib
PREFIX := tensor_scope

LIB := $(LIB_DIR)/lib$(PROJECT).so
CUR_DIR := $(shell pwd)

CXX ?=

CFLAGS := -std=c++17
LDFLAGS ?=

ifeq ($(DEBUG), 1)
	CFLAGS += -g -O0
else
	CFLAGS += -O3
endif

SRCS := $(notdir $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*/*.cpp))
OBJS := $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SRCS)))

TORCH_DIR = $(shell python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
PYTHON_INCLUDE_DIR = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB_DIR = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
PYTHON_VERSION = $(shell python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
INCLUDES := -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include -I$(PYTHON_INCLUDE_DIR)
LDFLAGS += -L$(TORCH_DIR)/lib -Wl,-rpath=$(TORCH_DIR)/lib -L$(PYTHON_LIB_DIR)/../ -Wl,-rpath=$(PYTHON_LIB_DIR)/../
LIBS := -lc10 -ltorch -ltorch_cpu -lpython$(PYTHON_VERSION)


all: dirs libs
dirs: $(OBJ_DIR) $(LIB_DIR)
libs: $(LIB)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

$(LIB): $(OBJS)
	$(CXX) $(LDFLAGS) -fPIC -shared -o $@ $^ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CFLAGS) -I$(INC_DIR) $(INCLUDES) -fPIC -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/*/%.cpp
	$(CXX) $(CFLAGS) -I$(INC_DIR) $(INCLUDES) -fPIC -c $< -o $@

.PHONY: clean
clean:
	-rm -rf $(OBJ_DIR) $(LIB_DIR) $(PREFIX)


.PHONY: install
install: all
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	cp -r $(LIB) $(PREFIX)/lib
	cp -r $(INC_DIR)/$(PROJECT).h $(PREFIX)/include
