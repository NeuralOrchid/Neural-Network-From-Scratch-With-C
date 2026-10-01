#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <SDL2/SDL.h>

// SDL Configuration
#define WINDOW_X 10
#define WINDOW_Y 10
#define WINDOW_WIDTH 1300
#define WINDOW_HEIGHT 700
#define NETWORK_WIDTH 1200
#define NETWORK_HEIGHT 600

// Inefficient way to draw a cyrcle
void SDL_RenderFillCircle(SDL_Renderer* renderer, float x, float y, float r) {
    for (int w = 0; w < 2 * r; w++) {
    for (int h = 0; h < 2 * r; h++) {
        int dx = r - w;
        int dy = r - h;
        if ( (dx*dx + dy*dy) <= (r*r) ) { SDL_RenderDrawPoint(renderer, x+dx, y+dy); }
    }}
}

// Architecture initialization
#define ARRAY_LEN(x) (sizeof(x) / sizeof(x[0]))
int arch[] = {2,5,5,1};

#define NUM_INPUTS 2
#define NUM_OUTPUTS 1
#define NUM_TRAINING_SETS 4

#define NUM_EPOCHS 200000
#define lr 0.1f

// Each data must be in [0, 1]
// Dataset (x)
double training_inputs[NUM_TRAINING_SETS][NUM_INPUTS] = {{0.0f,0.0f},
                                                        {1.0f,0.0f},
                                                        {0.0f,1.0f},
                                                        {1.0f,1.0f}};
// Dataset (y)
double training_outputs[NUM_TRAINING_SETS][NUM_OUTPUTS] = {{0.0f},
                                                            {1.0f},
                                                            {1.0f},
                                                            {0.0f}};

// Activation function and its derivative
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }

// Initialize weights
double init_weight() { return ((double)rand())/((double)RAND_MAX); }

// Shuffle the dataset
void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

// Linear neural network
struct LinearNN{
    size_t numInputs;
    size_t numOutputs;
    double* bias;
    double** weights;
    double* out;
    double* delta;
};

// Initializing weights and biases
void initLinearNN(struct LinearNN * inst, size_t numInputs, size_t numOutputs) {
    inst->numInputs = numInputs;
    inst->numOutputs = numOutputs;

    inst->delta = (double*)malloc(numOutputs * sizeof(double));
    inst->out = (double*)malloc(numOutputs * sizeof(double));
    inst->bias = (double*)malloc(numOutputs * sizeof(double));
    inst->weights = (double**)malloc(numOutputs * sizeof(double*));

    for (int i = 0; i < numOutputs; i++) {
        inst->weights[i] = (double*)malloc(numInputs * sizeof(double));

        for(int j=0; j < numInputs; j++){
          inst->weights[i][j] = init_weight();
        }
        inst->bias[i] = init_weight();
    }
}

// Forward
void forward(struct LinearNN* inst, double* x) {
    for (int i=0; i<inst->numOutputs; i++) {
        double activation = inst->bias[i];
        for (int j=0; j<inst->numInputs; j++) {
            activation += x[j] * inst->weights[i][j];
        }
        inst->out[i] = sigmoid(activation);
    }
}

// Backward
void backward(struct LinearNN* inst, double* subsequentDelta, double** subsequentWeights, size_t numSubsequentNodes) {
    for (int i=0; i<inst->numOutputs; i++) {
        double err = 0.0f;
        for (int j=0; j<numSubsequentNodes; j++) {
            err += subsequentDelta[j] * subsequentWeights[j][i];
        }
        inst->delta[i] = err * dSigmoid(inst->out[i]);
    }
}

// Stochastic gradient descent
void gradStep(struct LinearNN* inst, double* inputs) {
    for (int i=0; i<inst->numOutputs; i++) {
        inst->bias[i] += inst->delta[i] * lr;
        for(int j=0; j<inst->numInputs; j++) {
            inst->weights[i][j] += inputs[j] * inst->delta[i] * lr;
        }
    }
}

// Free memory
void freeLinearNN(struct LinearNN * inst) {
    for(int i=0; i < inst->numOutputs; i++) { free(inst->weights[i]); }
    free(inst->weights);
    free(inst->bias);
    free(inst->out);
    free(inst->delta);
}

int main(int argc, char *argv[]) {
    // SDL Window & Renderer
    SDL_Window* window;
    SDL_Renderer* renderer;

    // Check if there is any issue
    if ( SDL_INIT_VIDEO < 0 ) { fprintf(stderr, "ERROR: SDL_INIT_VIDEO"); }

    // Configure window
    window = SDL_CreateWindow(
        "",
        WINDOW_X,
        WINDOW_Y,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        SDL_WINDOW_BORDERLESS
    );

    // Check if there is any issue
    if ( !window ) { fprintf(stderr, "ERROR: !window"); }

    // Configure renderer
    renderer = SDL_CreateRenderer(
        window,
        -1,
        SDL_RENDERER_ACCELERATED
    );

    // Check if there is any issue
    if ( !renderer ) { fprintf(stderr, "ERROR: !renderer"); }

    // Calculating inner box position
    float net_x = (WINDOW_WIDTH / 2) - (NETWORK_WIDTH / 2);
    float net_y = (WINDOW_HEIGHT / 2) - (NETWORK_HEIGHT / 2);

    // Loss variable
    double loss[NUM_OUTPUTS];

    // Weight matrix of ones 
    double** ones = (double**)malloc(NUM_OUTPUTS * sizeof(double*));;
    for (int i=0; i < NUM_OUTPUTS; i++) {
        ones[i] = (double*)malloc(NUM_OUTPUTS * sizeof(double));
        for (int j=0; j < NUM_OUTPUTS; j++) {
            ones[i][j] = 1;
        }
    }

    // Linear layers
    size_t numLinearLayers = ARRAY_LEN(arch) - 1;
    struct LinearNN linearLayers[numLinearLayers];
    
    // Initializing weights and biases of layers
    for (int l = 0; l < numLinearLayers; l++) {
        initLinearNN(&linearLayers[l], arch[l], arch[l+1]);
    }

    // Dataset order
    int trainingSetOrder[] = {0,1,2,3};

    // SDL main loop
    SDL_Event event;
    bool quit = false;
    int epoch = 0;
    SDL_Color color;
    while( !quit ) {
        // Poll event ...
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_QUIT:
                quit = true;
                break;
            case SDL_KEYUP:
                break;
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                case SDLK_ESCAPE:
                    quit = true;
                    break;
                }
                break;
            }
        }
        // START Training loop

        if (epoch % NUM_TRAINING_SETS == 0) {
            // Shuffle the order of the training set
            shuffle(trainingSetOrder, NUM_TRAINING_SETS);
        }

        // Go through each of the training set elements
        int i = trainingSetOrder[epoch % NUM_TRAINING_SETS];

        // Compute forward pass
        forward(&linearLayers[0], training_inputs[i]);
        for (int l = 1; l < numLinearLayers; l++) {
            forward(&linearLayers[l], linearLayers[l-1].out);
        }

        // Print the results from forward pass
        printf ("Epoch[%d/%d]:    Input: %g %g    Output: %g    Expected: %g\n", 
                epoch, NUM_EPOCHS, 
                training_inputs[i][0], training_inputs[i][1],
                linearLayers[numLinearLayers-1].out[0], training_outputs[i][0]);

        // Backpropagation (if training...)
        if (epoch < NUM_EPOCHS) {
            // Compute loss
            for (int j=0; j < NUM_OUTPUTS; j++) {
                loss[j] = (training_outputs[i][j] - linearLayers[numLinearLayers-1].out[j]);
            }

            // Backward pass
            backward(
                &linearLayers[numLinearLayers-1],
                loss,
                ones,
                NUM_OUTPUTS
            );

            for (int l = 1; l < numLinearLayers; l++) {
                backward(
                    &linearLayers[numLinearLayers-l-1],
                    linearLayers[numLinearLayers-l].delta,
                    linearLayers[numLinearLayers-l].weights,
                    linearLayers[numLinearLayers-l].numOutputs
                );
            }

            // Apply stochastic gradient descent
            for (int l = 1; l < numLinearLayers; l++) {
                gradStep(
                    &linearLayers[numLinearLayers-l],
                    linearLayers[numLinearLayers-l-1].out
                );
            }
            gradStep(&linearLayers[0], training_inputs[i]);
        } else {
            // Small delay at the end (only if training was done)
            SDL_Delay(500);
        }

        // --- Visualization ---
        if ((epoch > NUM_EPOCHS) || (epoch % 100 == 0)) {
            SDL_RenderClear( renderer );
            // --- START Render Loop ---

            SDL_SetRenderDrawColor(renderer, 0x55, 0x55, 0x55, 0xFF);

            SDL_Rect box;
            box.w = NETWORK_WIDTH;
            box.h = NETWORK_HEIGHT;
            box.x = net_x;
            box.y = net_y;

            SDL_RenderDrawRect(renderer, &box);

            size_t num_layers = ARRAY_LEN(arch);
            float layer_width = NETWORK_WIDTH / num_layers;
            float x_offset = layer_width / 2;

            for (int l = 0; l < num_layers; l++) {
                float xPos = net_x + (l * layer_width) + x_offset;
                
                for (int j = 0; j < arch[l]; j++) {
                    float neuron_height = NETWORK_HEIGHT / arch[l];
                    float y_offset = neuron_height / 2;
                    float yPos = net_y + (j * neuron_height) + y_offset;

                    if (l < num_layers - 1) {
                        for (int k = 0; k < arch[l+1]; k++) {
                            float xPos2 = net_x + ((l+1) * layer_width) + x_offset;

                            float neuron_height2 = NETWORK_HEIGHT / arch[l+1];
                            float y_offset2 = neuron_height2 / 2;
                            float yPos2 = net_y + (k * neuron_height2) + y_offset2;
                            
                            if (linearLayers[l].weights[k][j] > 0) {
                                color.r = 0;
                                color.b = linearLayers[l].weights[k][j]*20;
                            } else {
                                color.r = linearLayers[l].weights[k][j]*(-20);
                                color.b = 0;
                            }
                            color.g = 0;

                            SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 0xFF);
                            SDL_RenderDrawLine(renderer, xPos, yPos, xPos2, yPos2);
                        }
                    }

                    float radius = 80 / arch[l] + 2;
                    if ( l == 0 ) {
                        color.r = training_inputs[i][j]*255;
                        color.g = training_inputs[i][j]*255;
                        color.b = training_inputs[i][j]*255;
                    } else {
                        color.r = linearLayers[l-1].out[j]*255;
                        color.g = linearLayers[l-1].out[j]*255;
                        color.b = linearLayers[l-1].out[j]*255;
                    }
                    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 0xFF);
                    SDL_RenderFillCircle(renderer, xPos, yPos, radius);
                }
            }

            // --- END Render Loop ---
            SDL_SetRenderDrawColor(renderer, 0x11, 0x11, 0x11, 0xFF);
            SDL_RenderPresent( renderer );
        }
        epoch++;
    }

    // Destroy SDL
    SDL_DestroyRenderer( renderer );
    SDL_DestroyWindow( window );
    SDL_Quit();

    // Free memory
    for (int l = 0; l < numLinearLayers; l++) { freeLinearNN(&linearLayers[l]); }
    for(int i=0; i < NUM_OUTPUTS; i++) { free(ones[i]); }
    free(ones);

    return EXIT_SUCCESS;
}
