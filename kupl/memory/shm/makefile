CC = mpicc

OBJS = shm.cpp
TARGET = shm

LDFLAGS = -lkupl

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	mpirun -np 2 -x UCX_TLS=sm ./$(TARGET)

clean:
	rm -f $(TARGET)