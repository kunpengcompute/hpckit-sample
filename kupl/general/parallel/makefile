CC = clang++

OBJS = parallel.cpp
TARGET = parallel

LDFLAGS = -lkupl

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ -O3 $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-7 ./$(TARGET)

clean:
	rm -f $(TARGET)