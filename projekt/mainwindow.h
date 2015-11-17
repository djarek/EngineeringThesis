#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <SDL2/SDL.h>
#include <memory>
#include "channel.h"
#include "typedefs.h"
#include <atomic>
#include <bits/stl_algo.h>

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
	uint cells;
	uint pixels_per_cell;
	ScalarField field;
	Channel_ptr<ScalarField> to_ui;
	Channel_ptr<ScalarField> from_ui;
public:
	MainWindow(int size_x, int size_y, uint cells, Channel_ptr<ScalarField> to_ui, Channel_ptr<ScalarField> from_ui):
		window(SDL_CreateWindow("Window", 0, 0, size_x, size_y, SDL_WINDOW_SHOWN)),
		renderer(SDL_CreateRenderer(window.get(), -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC)),
		cells(cells),
		pixels_per_cell(std::min(size_x, size_y) / cells),
		to_ui(to_ui),
		from_ui(from_ui)
	{}
	
	void event_loop() {
		SDL_Event event;
		bool quit = false;
		extern std::atomic<bool> running;
		while (!quit) {
			if (SDL_WaitEventTimeout(&event, 16)) {
				switch (event.type) {
				case SDL_QUIT:
					quit = true;
					running.store(false, std::memory_order_relaxed);
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
		auto renderer = this->renderer.get();
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		SDL_RenderClear(renderer);
		SDL_Rect rect;
		rect.w = pixels_per_cell;
		rect.h = pixels_per_cell;

		{
			ScalarField field;
			if (to_ui->try_pop(field)) {
				if (not this->field.empty()) {
					from_ui->try_push(this->field);
				}
				if (not field.empty()) {
					this->field = std::move(field);
				}
				
			}
		}
		
		if (not field.empty()) {
			for (uint y = 1; y < cells - 2; ++y) {
				for (uint x = 1; x < cells - 2; ++x) {
					rect.x = x * pixels_per_cell;
					rect.y = y * pixels_per_cell;
					SDL_SetRenderDrawColor(renderer, std::min(std::abs(255*field[y * cells + x]), 255.0f), 0, 0, 255);
					//std::cout << field[y * cells + x] << ' ';
					SDL_RenderFillRect(renderer, &rect);
				}
				//std::cout << std::endl;
			}
		}

		SDL_RenderPresent(renderer);
	}
};
#endif //MAINWINDOW_H
