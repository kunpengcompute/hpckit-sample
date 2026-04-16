CC = clang++

OBJS = parallel_for_3d.cpp
TARGET = parallel_for_3d

LDFLAGS = -lkupl

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-7 ./$(TARGET)

clean:
	rm -f $(TARGET)