package main

/*
#include <malloc.h>
#include <stdio.h>

void test_mallopt() {
	printf("Attempting to set M_MMAP_THRESHOLD in glibc...\n");
    // M_MMAP_THRESHOLD is glibc-specific and may not be supported by musl
    if (mallopt(M_MMAP_THRESHOLD, 64 * 1024) == 0) {
        printf("Failed to set M_MMAP_THRESHOLD (glibc-specific)\n");
    } else {
        printf("Successfully set M_MMAP_THRESHOLD\n");
    }
	printf("Okey");
}
*/
import "C"

import "fmt"

func main() {
	fmt.Println("Testing mallopt function...")
	C.test_mallopt() // This should fail on Alpine Linux
	fmt.Println("Done")
}
