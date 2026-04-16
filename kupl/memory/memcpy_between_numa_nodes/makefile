CC = clang++

OBJS = memcpy_between_numa_nodes.cpp
TARGET = memcpy_between_numa_nodes

LDFLAGS = -lkupl -lnuma

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	sh run.sh

clean:
	rm -f $(TARGET)
