.PHONY: run
run: neural
	clear && python3 teach_squares.py | ./neural 2 14 1

neural: neural.c
	gcc -o neural neural.c --std=c11 -Wall -Wextra
