CC?=gcc
CFLAGS?=-O2 -Wall -Wextra
LDFLAGS?=-static  -I../include
AR?=ar

.PHONY: all clean

SRC=\
	arrays.c \
	time.c

OBJ=$(SRC:.c=.o)

all: libbenchmark.a

$(OBJ): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

libbenchmark.a: $(OBJ)
	$(AR) rcs $@ $^

clean:
	rm -f $(OBJ) libbenchmark.a
