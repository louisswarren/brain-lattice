#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

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

/*
    ---------              |--------|              ----------
	| Input | -----------> | Hidden | -----------> | Output |
	---------              |--------|              ----------
	 neurons[0]             neurons[1]              neurons[2]
	           weights[0]              weights[1]
	           biases[0]               biases[1]
	                        errors[0]               errors[1]
*/


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
	return A;
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
		printf("% 08lf  ", v->elem[i]);
		if (i == v->len - 1)
			printf("]]\n");
		else
			printf("],\n");
	}
}

void print_matrix(Colmatrix *A)
{
	for (size_t i = 0; i < A->rows; ++i) {
		if (i == 0)
			printf("[[ ");
		else
			printf(" [ ");
		forindex(j, A->row[i]) {
			printf("% 08lf", A->row[i]->elem[j]);
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

	memcpy(brain->widths, widths, depth * sizeof(widths[0]));

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



void check(Brain *brain, Vector *expected)
{
	think(brain);
	printf("Computed %lf -> %lf\n",
			brain->neurons[0]->elem[0], brain->neurons[1]->elem[0]);
	int d = brain->depth - 1;

	forindex(k, brain->errors[d - 1]) {
		double out = brain->neurons[d]->elem[k];
		double target = expected->elem[k];
		brain->errors[d - 1]->elem[k] = out * (1 - out) * (target - out);
		printf("Error term: %lf * %lf * %lf = %lf\n",
				out, 1 - out, target - out, brain->errors[d - 1]->elem[k]);
	}


	for (--d; d > 0; --d) {
		forindex(h, brain->errors[d - 1]) {
			double out = brain->neurons[d]->elem[h];
			double errsum = 0;
			forindex(k, brain->errors[d]) {
				double effect_weight = weightfromto(brain, d, h, k);
				double effect_error = brain->errors[d]->elem[k];
				errsum += effect_weight * effect_error;
			}
			brain->errors[d - 1]->elem[h] = out * (1 - out) * errsum;
		}
	}
}

void learn(Brain *brain, Vector *expected)
{
	check(brain, expected);

	for (size_t d = 0; d < brain->depth - 1; ++d) {
		forindex(j, brain->neurons[d + 1]) {
			double error = brain->errors[d]->elem[j];
			forindex(i, brain->neurons[d]) {
				double value = brain->neurons[d]->elem[i];
				weightfromto(brain, d, i, j) += brain->rate * error * value;
				printf("Weight update: %lf * %lf * %lf = %lf\n",
						brain->rate, error, value, brain->rate * error * value);
			}
			brain->biases[d]->elem[j] += brain->rate * error;
			printf("Bias update: %lf * %lf = %lf\n",
					brain->rate, error, brain->rate * error);
		}
	}
}

void learn_loop(Brain *brain)
{
	double x;
	int cycle_pos = 0;
	int input_len = brain->widths[0];
	int output_len = brain->widths[brain->depth - 1];
	Vector *expected = new_vector(output_len);

	printf("y = %lf x + %lf\n", brain->weights[0]->row[0]->elem[0], brain->biases[0]->elem[0]);

	while (scanf("%lf", &x) >= 1) {
		if (cycle_pos < input_len)
			brain->neurons[0]->elem[cycle_pos] = x;
		else
			expected->elem[cycle_pos - input_len] = x;
		cycle_pos++;
		if (cycle_pos == input_len + output_len) {
			cycle_pos = 0;
			puts(""); puts(""); puts(""); puts("");
			printf("Learning from %lf -> %lf\n",
					brain->neurons[0]->elem[0], expected->elem[0]);
			learn(brain, expected);
			printf("y = %lf x + %lf\n", brain->weights[0]->row[0]->elem[0], brain->biases[0]->elem[0]);
		}
	}
	free(expected);
	printf("Done learning.\n");
}

void classify_loop(Brain *brain)
{
	double x;
	int cycle_pos = 0;
	int input_len = brain->widths[0];

	while (scanf("%lf", &x) >= 1) {
		brain->neurons[0]->elem[cycle_pos] = x;
		cycle_pos++;
		printf("%lf ", x);
		if (cycle_pos == input_len) {
			cycle_pos = 0;
			think(brain);

			forindex(i, brain->output)
				printf("%lf ", brain->output->elem[i]);
			printf("\n");
		}
	}
}


void usage(char *prog)
{
	fprintf(stderr, "Usage: %s [-r learning_rate] w1 w2 [w3 ...]\n", prog);
	fprintf(stderr, "       %s -f knowledge_file\n", prog);
	fprintf(stderr, "\n");
	fprintf(stderr, "Learning mode:\n");
	fprintf(stderr, "    w1 ... wn are the widths of each neural layer.\n");
	fprintf(stderr, "    Input is whitespace-separated floats on stdin.\n");
	fprintf(stderr, "    Output is the weights and biases learned.\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Classifying mode:\n");
	fprintf(stderr, "    knowledge_file contains weights and biases.\n");
	fprintf(stderr, "    Input and output is whitespace-separated.\n");
}

int main(int argc, char **argv)
{
	double learning_rate = 0.05;
	size_t depth;
	size_t *widths;
	int learn_mode = 1;
	int argerror = 0;

	Brain *brain;

	int c;
	if ((c = getopt(argc, argv, "f:r:")) != -1) {
		switch (c) {
		case 'f':
			learn_mode = 0;
			break;
		case 'r':
			learning_rate = atof(optarg);
			break;
		default:
			argerror = 1;
		}
	}

	argerror += (learn_mode && argc <= optind + 1);


	if (learn_mode && !argerror) {
		depth = argc - optind;
		widths = emalloc(sizeof(widths[0]) * depth);

		size_t offset = optind;
		for (size_t i = 0; i < depth; ++i) {
			widths[i] = atoi(argv[i + offset]);
		}

		brain = new_brain(depth, widths, learning_rate);
//		brain->weights[0]->row[0]->elem[0] = 0;
//		brain->biases[0]->elem[0] = -0.5;
		learn_loop(brain);
		classify_loop(brain);
	} else if (!learn_mode) {
		// Load brain from file and start classifying
	}

	if (argerror) {
		usage(argv[0]);
	}

	return 0;
}
