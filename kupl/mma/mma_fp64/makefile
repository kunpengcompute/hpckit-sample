CC = clang++

OBJS = mma_fp64.cpp
TARGET = mma_fp64

LDFLAGS = -lkupl

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ -O3 $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	taskset -c 0 ./$(TARGET)

clean:
	rm -f $(TARGET)