#include <stdlib.h>
#include <stdio.h>

struct neuron {
	size_t n;
	double **inputs;
	double output;
	double bias;
	double *weights;
};

struct brain {
	struct neuron **neurons;
	size_t len;
	double *inputs;
	size_t input_len;
	double **outputs;
	size_t output_len;
};

struct neuron *new_neuron(size_t n)
{
	struct neuron *neuron = malloc(sizeof(*neuron));
	double **inputs = malloc(n * sizeof(*inputs));
	double *weights = malloc(n * sizeof(*weights));

	neuron->n = n;
	neuron->inputs = inputs;
	neuron->weights = weights;
}

void delete_neuron(struct neuron *neuron)
{
	free(neuron->inputs);
	free(neuron->weights);
	free(neuron);
}

struct brain *brain_lattice(size_t n, size_t depth, size_t *layer_sizes)
{
	struct brain *brain = malloc(sizeof(brain));
	size_t neuron_counter = 0;

	brain->inputs = malloc(n * sizeof(double));
	brain->input_len = n;
	brain->outputs = malloc(layer_sizes[depth - 1] * sizeof(double *));
	brain->output_len = layer_sizes[depth - 1];

	brain->len = 0;
	for (size_t i = 0; i < depth; ++i)
		brain->len += layer_sizes[i];
	brain->neurons = malloc(brain->len * sizeof(struct neuron *));

	/* Connect the first layer of neurons to the inputs. */
	for (size_t i = 0; i < layer_sizes[0]; ++i) {
		brain->neurons[neuron_counter] = new_neuron(n);
		for (size_t j = 0; j < n; ++j)
			brain->neurons[neuron_counter]->inputs[j] = &brain->inputs[j];
		neuron_counter++;
	}

	/* Connect the remaining layers to the previous layers. */
	for (size_t d = 1; d < depth; ++d) {
		for (size_t i = 0; i < layer_sizes[d]; ++i) {
			brain->neurons[neuron_counter] = new_neuron(layer_sizes[d - 1]);
			for (size_t j = 0; j < n; ++j) {
				size_t idx = neuron_counter - i - layer_sizes[d - 1] + j;
				double *output = &brain->neurons[idx]->output;
				brain->neurons[neuron_counter]->inputs[j] = output;
			}
			neuron_counter++;
		}
	}

	return brain;
}

void delete_brain(struct brain *brain)
{
	for (size_t i = 0; i < brain->len; ++i)
		delete_neuron(brain->neurons[i]);
	free(brain->inputs);
	free(brain->outputs);
}

void neuron_think(struct neuron *neuron)
{
	neuron->output = neuron->bias;
	for (size_t i = 0; i < neuron->n; ++i)
		neuron->output += neuron->weights[i] * *neuron->inputs[i];
}

int main(void)
{
	size_t widths[] = {2, 3, 1};
	struct brain *brain = brain_lattice(2, 3, widths);

	struct neuron *neuron = brain->neurons[0];
	neuron_think(neuron);

	printf("%f\n", neuron->output);
}
