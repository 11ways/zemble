CC = gcc
all: point
point: point.o
	$(CC) -o point point.o
.PHONY: clean
clean:
	rm -f point
