CC = clang++

OBJS = mma_bf16.cpp
TARGET = mma_bf16

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