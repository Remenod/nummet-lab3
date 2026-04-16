CC = gcc
CXX = g++
CFLAGS = -O2 -Wall -Wextra
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra

TARGET = app
BUILD_DIR = build

C_SRC = lib/tinyexpr.c
CPP_SRC = main.cpp

OBJS = $(BUILD_DIR)/$(notdir $(C_SRC:.c=.o)) $(BUILD_DIR)/$(notdir $(CPP_SRC:.cpp=.o))

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/tinyexpr.o: $(C_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Wno-array-bounds -c $< -o $@

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run