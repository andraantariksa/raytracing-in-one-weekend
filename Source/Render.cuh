void CUDA_render_setup(int width, int height);
void CUDA_render_destroy();
void CUDA_render_render(int windowWidth, int windowHeight, int pixelSamples, Camera camera, const HittableObjects& world, Surface& s);
