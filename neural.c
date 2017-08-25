#include <stdlib.h>
#include <stdio.h>

#define mat(A, I, J) A->x[I * A->m + J]

struct vector {
	size_t n;
	double x[];
};

struct matrix {
	size_t n, m;
	double x[];
};

struct brain {
	size_t depth;
	size_t *layer_sizes;
	struct matrix *weights;
};

struct vector *new_vector(size_t n)
{
	struct vector *v = malloc(sizeof(*v) + n * sizeof(v->x[0]));
	if (v)
		v->n = n;
	return v;
}

struct matrix *new_matrix(size_t n, size_t m)
{
	struct matrix *a = malloc(sizeof(*a) + n * m * sizeof(a->x[0]));
	if (a) {
		a->n = n;
		a->m = m;
	}
	return a;
}

void print_vector(struct vector *v)
{
	for (size_t i = 0; i < v->n; ++i) {
		printf("| ");
		printf("%8f ", v->x[i]);
		printf("|\n");
	}
}

void print_matrix(struct matrix *a)
{
	for (size_t row = 0; row < a->n; ++row) {
		printf("| ");
		for (size_t col = 0; col < a->m; ++col)
			printf("%8f ", mat(a, row, col));
		printf("|\n");
	}
}

int main(void)
{
	struct vector *v = new_vector(3);
	v->x[0] = 1;
	v->x[1] = 3;
	v->x[2] = 4;
	print_vector(v);
	free(v);
	struct matrix *a = new_matrix(2, 3);
	mat(a, 0, 0) = 1;
	mat(a, 0, 1) = 2;
	mat(a, 0, 2) = 3;
	mat(a, 1, 0) = 4;
	mat(a, 1, 1) = 5;
	mat(a, 1, 2) = 6;
	print_matrix(a);
}
