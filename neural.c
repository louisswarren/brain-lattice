#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define forindex(I, V) for (size_t I = 0; I < V->len; ++I)
#define allocate(X, N) X = emalloc(sizeof(*X) * N)
#define weightfromto(B, D, I, J) B->weights[D]->row[J]->elem[I]

void *emalloc(size_t size)
{
	void *x = malloc(size);
	if (!x) {
		fprintf(stderr, "Out of memory.\n");
		exit(EXIT_FAILURE);
	}
	return x;
}

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
	double rate;
	Vector **neurons;
	Colmatrix **weights;
	Vector **biases;
	Vector **errors;
	Vector *output;
} Brain;


/* Misc maths */

double rand_weight()
{
	return (rand() / (double)(RAND_MAX)) * 0.1 - 0.05;
}

double sigmoid_approx(double x)
{
	return x / (1 + fabs(x));
}


/* Linear algebra functions */

Vector *new_vector(size_t len)
{
	Vector *v = emalloc(sizeof(*v) + len * sizeof(v->elem[0]));
	v->len = len;
	forindex(i, v)
		v->elem[i] = rand_weight();
	return v;
}

Colmatrix *new_colmatrix(size_t rows, size_t cols)
{
	Colmatrix *A = emalloc(sizeof(*A) + rows * sizeof(A->row[0]));
	for (size_t i = 0; i < rows; ++i) {
		A->row[i] = new_vector(cols);
	}
	A->rows = rows;
	A->cols = cols;
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

void print_vector(Vector *v)
{
	forindex(i, v) {
		if (i == 0)
			printf("[[ ");
		else
			printf(" [ ");
		printf("% 08f  ", v->elem[i]);
		if (i == v->len - 1)
			printf("]]\n");
		else
			printf("],\n");
	}
	printf("\n");
}

void print_matrix(Colmatrix *A)
{
	for (size_t i = 0; i < A->rows; ++i) {
		if (i == 0)
			printf("[[ ");
		else
			printf(" [ ");
		forindex(j, A->row[i]) {
			printf("% 08f", A->row[i]->elem[j]);
			if (j == A->row[i]->len - 1)
				printf("  ");
			else
				printf(", ");
		}
		if (i == A->rows - 1)
			printf("]]\n");
		else
			printf("],\n");
	}
	printf("\n");
}


/* Neural network functions */

Brain *new_brain(size_t depth, const size_t widths[], double rate)
{
	Brain *brain = emalloc(sizeof(*brain));

	brain->depth = depth;
	brain->rate = rate;

	allocate(brain->widths, depth);
	allocate(brain->neurons, depth);
	allocate(brain->weights, depth - 1);
	allocate(brain->biases, depth - 1);
	allocate(brain->errors, depth - 1);

	memcpy(brain->widths, widths, depth);

	for (size_t i = 0; i < depth; ++i) {
		brain->neurons[i] = new_vector(widths[i]);
		if (i < depth - 1) {
			brain->weights[i] = new_colmatrix(widths[i + 1], widths[i]);
			brain->biases[i] = new_vector(widths[i + 1]);
			brain->errors[i] = new_vector(widths[i + 1]);
		}
	}

	brain->output = brain->neurons[depth - 1];

	return brain;
}

/* Input is b->neurons[0], output is b->neurons[b->depth] */
void think(Brain *b)
{
	for (size_t k = 0; k < b->depth - 1; ++k) {
		transformation(b->neurons[k + 1], b->weights[k], b->neurons[k]);
		shift(b->neurons[k + 1], b->biases[k]);
	}
	forindex(k, b->neurons[b->depth - 1])
		b->output->elem[k] = sigmoid_approx(b->output->elem[k]);
}

void think_about(Brain *brain, Vector *v)
{
	assert(v->len == brain->neurons[0]->len);
	memcpy(brain->neurons[0]->elem, v->elem, sizeof(v->elem[0]) * v->len);
	think(brain);
}

void check(Brain *brain, Vector *input, Vector *expected)
{
	size_t d = brain->depth - 1;

	forindex(k, brain->errors[d]) {
		double out = brain->neurons[d]->elem[k];
		double target = expected->elem[k];
		brain->errors[d-1]->elem[k] = out * (1 - out) * (target - out);
	}

	for (; d >= 0; --d) {
		forindex(k, brain->errors[d]) {
			double out = brain->neurons[d]->elem[k];
			double errsum = 0;
			forindex(n, brain->errors[d + 1]) {
				double effect_weight = weightfromto(brain, d, k, n);
				double effect_error = brain->errors[d]->elem[n];
				errsum += effect_weight * effect_error;
			}
			brain->errors[d-1]->elem[k] = out * (1 - out) * errsum;
		}
	}
}

void learn(Brain *brain, Vector *input, Vector *output)
{
	check(brain, input, output);
	for (size_t d = 0; d < brain->depth - 1; --d) {
		forindex(i, brain->neurons[d]) {
			forindex(j, brain->neurons[d + 1]) {
				double error = brain->errors[d]->elem[j];
				double value = brain->neurons[d]->elem[i];
				weightfromto(brain, d, i, j) += brain->rate * error * value;
			}
		}
	}
}

int main(void)
{
	size_t widths[] = {3, 4, 2};
	Brain *brain = new_brain(3, widths, 0.05);
	think(brain);

	print_vector(brain->neurons[0]);

	puts("\n-->\n");

	print_matrix(brain->weights[0]);
	puts("*");
	print_vector(brain->neurons[0]);
	puts("+");
	print_vector(brain->biases[0]);
	puts("=");
	print_vector(brain->neurons[1]);

	puts("\n-->\n");

	print_matrix(brain->weights[1]);
	puts("*");
	print_vector(brain->neurons[1]);
	puts("+");
	print_vector(brain->biases[1]);
	puts("=");
	print_vector(brain->neurons[2]);
	return 0;
}
