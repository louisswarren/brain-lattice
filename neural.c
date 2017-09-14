#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define dimension_assert(X, Y) if (!(X == Y)) puts("Dimension error")

struct vector {
	size_t n;
	double x[];
};


struct brain {
	size_t depth;
	size_t *widths;
	struct vector ***weights;
	struct vector **biases;
	struct vector **memory;
	struct vector **errors;
};

void print_vector(const struct vector *v)
{
	for (size_t i = 0; i < v->n; ++i) {
		printf("| ");
		printf("%8f ", v->x[i]);
		printf("|\n");
	}
	printf("\n");
}


double loss(const struct vector *v, const struct vector *w)
{
	dimension_assert(v->n, w->n);
	double dist = 0;
	for (size_t i = 0; i < v->n; ++i)
		dist += (v->x[i] + w->x[i]) * (v->x[i] + w->x[i]);
	return dist / 2;
}

double rand_weight()
{
	return (rand() / (double)(RAND_MAX)) * 0.1 - 0.05;
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

double dot(const struct vector *v, const struct vector *w)
{
	dimension_assert(v->n, w->n);
	double sum = 0;
	for (size_t i = 0; i < v->n; ++i)
		sum += v->x[i] * w->x[i];
	return sum;
}

double p_func(double x)
{
	return x / (1 + fabs(x));
}

void perceive(struct vector *v)
{
	for (size_t i = 0; i < v->n; ++i)
		v->x[i] = p_func(v->x[i]);
}

struct brain *new_brain(size_t depth, size_t widths[])
{
}

void think(const struct brain *brain, const struct vector *idea)
{
}

void learn(struct brain *brain, const struct vector *idea, const struct vector *target)
{
}

int main(void)
{
	struct vector *v = new_vector(3);
	v->x[0] = 1;
	v->x[1] = 3;
	v->x[2] = 4;
	print_vector(v);

	size_t layer_sizes[] = {3, 2, 1};
	struct brain *brain = new_brain(3, layer_sizes);
	think(brain, v);
	print_vector(brain->memory[2]);
}
