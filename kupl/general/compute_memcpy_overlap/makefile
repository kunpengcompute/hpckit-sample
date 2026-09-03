CC = clang++

OBJS = compute_memcpy_overlap.cpp
TARGET = compute_memcpy_overlap

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