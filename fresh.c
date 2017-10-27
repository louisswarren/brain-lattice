#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define forindex(I, V) for (size_t I = 0; I < V->len; ++I)
#define allocate(X, N) X = malloc(sizeof(*X) * N)

typedef struct {
	size_t len;
	double elem[];
} Vector;

typedef struct {
	size_t rows, cols;
	Vector *row[];
} Colmatrix;

typedef struct {
	size_t depth;
	size_t *widths;
	Vector **neurons;
	Colmatrix **weights;
	Vector **biases;
} Brain;


/* Linear algebra functions */

Vector *new_vector(size_t len)
{
	Vector *v = malloc(sizeof(*v) + len * sizeof(v->elem[0]));
	if (v)
		v->len = len;
	return v;
}

Colmatrix *new_colmatrix(size_t rows, size_t cols)
{
	Colmatrix *A = malloc(sizeof(*A) + rows * sizeof(A->row[0]));
	if (A) {
		A->rows = rows;
		A->cols = cols;
	}
}

double dot_product(const Vector *v, const Vector *w)
{
	assert(v->len == w->len);
	double x = 0;
	forindex(i, v)
		x += v->elem[i] * w->elem[i];
	return x;
}

void transformation(Vector *y, const Colmatrix *A, const Vector *x)
{
	assert(y->len == A->rows && A->cols == x->len);
	forindex(i, y)
		y->elem[i] = dot_product(A->row[i], x);
}

void shift(Vector *v, const Vector *w)
{
	assert(v->len == w->len);
	forindex(i, v)
		v->elem[i] += w->elem[i];
}



/* Neural network functions */

Brain *new_brain(size_t depth, const size_t widths[])
{
	Brain *brain = malloc(sizeof(*brain));

	if (!brain) return 0;

	brain->depth = depth;
	allocate(brain->widths, depth);
	memcpy(brain->widths, widths, depth);

	if (!(allocate(brain->neurons, depth)))     return 0;
	if (!(allocate(brain->weights, depth - 1))) return 0;
	if (!(allocate(brain->biases,  depth - 1))) return 0;


	for (size_t i = 0; i < depth; ++i) {
		brain->neurons[i] = new_vector(widths[i]);
		if (i < depth - 1) {
			brain->weights[i] = new_colmatrix(widths[i + 1], widths[i]);
			brain->biases[i] = new_vector(widths[i + 1]);
		}
	}
}

/* Input is b->neurons[0], output is b->neurons[b->depth] */
void think(Brain *b)
{
	for (size_t k = 0; k < b->depth - 1; ++k) {
		transformation(b->neurons[k + 1], b->weights[k], b->neurons[k]);
		shift(b->neurons[k + 1], b->biases[k]);
	}
}


int main(void)
{
	return 0;
}
