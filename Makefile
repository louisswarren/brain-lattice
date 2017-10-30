.PHONY: run
run: neural
	echo "1.23 4.56 78.901 0.2 0.8" | ./neural 3 4 2

neural: neural.c
	gcc -o neural neural.c --std=c11 -Wall -Wextra
