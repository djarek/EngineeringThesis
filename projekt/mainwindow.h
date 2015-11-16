#include <SDL2/SDL.h>
#include <memory>

struct DestroyWindow
{
	void operator()(SDL_Window* window) const {
		SDL_DestroyWindow(window);
	}
};

struct DestroyRenderer
{
	void operator()(SDL_Renderer* renderer) const {
		SDL_DestroyRenderer(renderer);
	}
};

class MainWindow
{
	std::unique_ptr<SDL_Window, DestroyWindow> window;
	std::unique_ptr<SDL_Renderer, DestroyRenderer> renderer;
public:
	MainWindow(int size_x, int size_y):
		window(SDL_CreateWindow("Window", 0, 0, size_x, size_y, SDL_WINDOW_SHOWN)),
		renderer(SDL_CreateRenderer(window.get(), -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC))
	{}
	
	void event_loop() {
		SDL_Event event;
		bool quit = false;
		while (!quit) {
			if (SDL_WaitEvent(&event)) {
				switch (event.type) {
				case SDL_QUIT:
					quit = true;
					break;
				case SDL_WINDOWEVENT:
					paint();
					continue;
				case SDL_MOUSEMOTION:
					//onMouseMove(event);
					break;
				case SDL_MOUSEBUTTONUP:
				case SDL_MOUSEBUTTONDOWN:
					//onMouseButtonClick(event);
					break;
				default:
					break;
				}
			}

			paint();
			SDL_Delay(33);
		}
	}
	
	void paint() {
		SDL_SetRenderDrawColor(renderer.get(), 255, 255, 255, 255);
		SDL_RenderClear(renderer.get());
		SDL_RenderPresent(renderer.get());
	}
};
