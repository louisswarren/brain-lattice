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
	struct matrix **weights;
	struct vector **biases;
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

struct matrix *vec_to_mat(struct vector *v)
{
	struct matrix *a = new_matrix(v->n, 1);
	if (a) {
		for (size_t i = 0; i < v->n; ++i)
			mat(a, i, 0) = v->x[i];
	}
	return a;
}

struct matrix *matrix_product(struct matrix *a, struct matrix *b)
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

struct brain *new_brain(size_t depth, size_t *layer_sizes)
{
	struct brain *brain = malloc(sizeof(*brain));
	brain->depth = depth;
	brain->layer_sizes = layer_sizes;
	brain->weights = malloc((depth - 1) * sizeof(*brain->weights));
	brain->biases = malloc((depth - 1) * sizeof(*brain->biases));

	for (size_t i = 0; i < depth - 1; ++i) {
		brain->weights[i] = new_matrix(layer_sizes[i + 1], layer_sizes[i]);
		brain->biases[i] = new_vector(layer_sizes[i + 1]);
	}
}

struct vector *think(struct brain *brain, struct vector *input)
{

}

void print_vector(struct vector *v)
{
	for (size_t i = 0; i < v->n; ++i) {
		printf("| ");
		printf("%8f ", v->x[i]);
		printf("|\n");
	}
	printf("\n");
}

void print_matrix(struct matrix *a)
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
	print_matrix(b);
	free(v);

	struct matrix *a = new_matrix(2, 3);
	mat(a, 0, 0) = 1;
	mat(a, 0, 1) = 2;
	mat(a, 0, 2) = 3;
	mat(a, 1, 0) = 4;
	mat(a, 1, 1) = 5;
	mat(a, 1, 2) = 6;
	print_matrix(a);

	struct matrix *c = matrix_product(a, b);
	print_matrix(c);

	struct matrix *b2 = new_matrix(2, 2);
	mat(b2, 0, 0) = -2;
	mat(b2, 0, 1) = 1;
	mat(b2, 1, 0) = 2;
	mat(b2, 1, 1) = 3;
	print_matrix(b2);

	struct matrix *d = matrix_product(b2, a);
	print_matrix(d);
}
