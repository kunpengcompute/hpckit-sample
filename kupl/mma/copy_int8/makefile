CC = clang++

OBJS = copy_int8.cpp
TARGET = copy_int8

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