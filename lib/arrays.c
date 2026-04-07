#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>

#include <benchmark/arrays.h>

void *benchmark_alloc_data(size_t n, size_t size) {
	void *ptr = nullptr;
	size_t alloc_sz = n * size;
	int err = posix_memalign(&ptr, 4096, alloc_sz);
	if (err != 0) {
		fprintf(stderr, "Error allocating memory: %d\n", err);
	}
	return ptr;
}