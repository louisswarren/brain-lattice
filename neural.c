#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
	struct matrix **weights;
	struct vector **biases;
};

double rand_weight()
{
	return (rand() / (double)(RAND_MAX)) * 2 - 1;
}

struct vector *new_vector(size_t n)
{
	struct vector *v = malloc(sizeof(*v) + n * sizeof(v->x[0]));
	if (v)
		v->n = n;
	return v;
}

struct vector *new_rand_vector(size_t n)
{
	struct vector *v = new_vector(n);
	if (v) {
		for (size_t i = 0; i < n; ++i)
			v->x[i] = rand_weight();
	}
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

struct matrix *new_rand_matrix(size_t n, size_t m)
{
	struct matrix *a = new_matrix(n, m);
	if (a) {
		for (size_t i = 0; i < n * m; ++i)
			a->x[i] = rand_weight();
	}
	return a;
}

struct matrix *vec_to_mat(const struct vector *v)
{
	struct matrix *a = new_matrix(v->n, 1);
	if (a) {
		for (size_t i = 0; i < v->n; ++i)
			mat(a, i, 0) = v->x[i];
	}
	return a;
}

struct matrix *matrix_product(const struct matrix *a, const struct matrix *b)
{
	struct matrix *c = new_matrix(a->n, b->m);
	if (c && a->m == b-> n) {
		for (size_t i = 0; i < c->n; ++i) {
			for (size_t j = 0; j < c->m; ++j) {
				mat(c, i, j) = 0;
				for (size_t k = 0; k < a->m; ++k)
					mat(c, i, j) += mat(a, i, k) * mat(b, k, j);
			}
		}
	}
	return c;
}

void multiply_vector(const struct matrix *a, struct vector **vp)
{
	struct vector *y = new_vector(a->n);
	if (y && a->m == (*vp)->n) {
		for (size_t i = 0; i < a->n; ++i) {
			y->x[i] = 0;
			for (size_t k = 0; k < a->m; ++k)
				y->x[i] += mat(a, i, k) * (*vp)->x[k];
		}
		free(*vp);
		*vp = y;
	}
}

void add_vector(const struct vector *summand, struct vector *v)
{
	for (size_t i = 0; i < v->n; ++i)
		v->x[i] += summand->x[i];
}

struct brain *new_brain(size_t depth, size_t layer_sizes[])
{
	struct brain *brain = malloc(sizeof(*brain));
	brain->depth = depth;
	brain->layer_sizes = malloc(depth * sizeof(brain->layer_sizes[0]));
	memcpy(brain->layer_sizes, layer_sizes, depth);
	brain->weights = malloc((depth - 1) * sizeof(*brain->weights));
	brain->biases = malloc((depth - 1) * sizeof(*brain->biases));

	for (size_t i = 0; i < depth - 1; ++i) {
		brain->weights[i] = new_rand_matrix(layer_sizes[i + 1], layer_sizes[i]);
		brain->biases[i] = new_rand_vector(layer_sizes[i + 1]);
	}
	return brain;
}

void think(const struct brain *brain, struct vector **idea)
{
	for (size_t d = 0; d < brain->depth - 1; ++d) {
		multiply_vector(brain->weights[d], idea);
		add_vector(brain->biases[d], *idea);
	}
}

void print_vector(const struct vector *v)
{
	for (size_t i = 0; i < v->n; ++i) {
		printf("| ");
		printf("%8f ", v->x[i]);
		printf("|\n");
	}
	printf("\n");
}

void print_matrix(const struct matrix *a)
{
	for (size_t row = 0; row < a->n; ++row) {
		printf("| ");
		for (size_t col = 0; col < a->m; ++col)
			printf("%8f ", mat(a, row, col));
		printf("|\n");
	}
	printf("\n");
}

int main(void)
{
	struct vector *v = new_vector(3);
	v->x[0] = 1;
	v->x[1] = 3;
	v->x[2] = 4;
	print_vector(v);

	struct matrix *b = vec_to_mat(v);

	struct matrix *a = new_matrix(2, 3);
	mat(a, 0, 0) = 1;
	mat(a, 0, 1) = 2;
	mat(a, 0, 2) = 3;
	mat(a, 1, 0) = 4;
	mat(a, 1, 1) = 5;
	mat(a, 1, 2) = 6;

	struct matrix *c = matrix_product(a, b);

	struct matrix *b2 = new_matrix(2, 2);
	mat(b2, 0, 0) = -2;
	mat(b2, 0, 1) = 1;
	mat(b2, 1, 0) = 2;
	mat(b2, 1, 1) = 3;

	struct matrix *d = matrix_product(b2, a);

	size_t layer_sizes[] = {3, 20, 1};
	struct brain *brain = new_brain(3, layer_sizes);
	think(brain, &v);
	print_vector(v);
}
