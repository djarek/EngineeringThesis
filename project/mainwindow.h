/**
 * FluidSim - a free and open-source interactive fluid flow simulator
 * Copyright (C) 2015  Damian Jarek <damian.jarek93@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <SDL2/SDL.h>
#include <memory>
#include "channel.h"
#include "typedefs.h"
#include <atomic>

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
	SDL_Rect boundary_rect;
	ScalarField field;
	Channel_ptr<ScalarField> dye_field_to_ui;
	Channel_ptr<Event> events_from_ui;
	bool left_mouse_button_pressed {false};
public:
	MainWindow(int size_x, int size_y, uint cells, Channel_ptr<ScalarField> dye_field_to_ui, Channel_ptr<Event> events_from_ui):
		window(SDL_CreateWindow("Window", 0, 0, size_x, size_y, SDL_WINDOW_SHOWN/* | SDL_WINDOW_FULLSCREEN*/)),
		renderer(SDL_CreateRenderer(window.get(), -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC)),
		cells(cells),
		pixels_per_cell(std::min(size_x, size_y) / cells),
		dye_field_to_ui(dye_field_to_ui),
		events_from_ui(events_from_ui)
	{
		boundary_rect.w = pixels_per_cell * cells;
		boundary_rect.h = pixels_per_cell * cells;
		boundary_rect.x = boundary_rect.y = 0;
		field.resize(cells * cells);
	}

	void onMouseButtonUp(const SDL_Event& event)
	{
		if (event.button.button == SDL_BUTTON_LEFT) {
			left_mouse_button_pressed = false;
		}
	}

	void onMouseButtonDown(const SDL_Event& event)
	{
		if (event.button.button == SDL_BUTTON_LEFT) {
			left_mouse_button_pressed = true;
		} else if (event.button.button == SDL_BUTTON_RIGHT) {
			Event simulation_event;
			simulation_event.point = Point{static_cast<cl_int>(1.0 * event.button.x / pixels_per_cell),
						       static_cast<cl_int>(1.0 * event.button.y / pixels_per_cell)};
			simulation_event.type = Event::Type::ADD_DYE;
			simulation_event.value.as_scalar = Scalar{1};
			events_from_ui->try_push(simulation_event);
		}
	}

	Vector normalize_velocity(float x, float y)
	{
		return Vector{std::max(std::min(x, 5.0f), -5.0f), std::max(std::min(y, 5.0f), -5.0f)};
	}

	void onMouseMove(const SDL_Event& event)
	{
		if (left_mouse_button_pressed) {
			Event simulation_event;

			simulation_event.point = Point{static_cast<cl_int>(1.0 * event.motion.x / pixels_per_cell),
						       static_cast<cl_int>(1.0 * event.motion.y / pixels_per_cell)};
			simulation_event.value.as_vector = normalize_velocity(1.0f * event.motion.xrel / pixels_per_cell,
									      1.0f * event.motion.yrel / pixels_per_cell);

			simulation_event.type = Event::Type::APPLY_FORCE;
			events_from_ui->try_push(simulation_event);
		}
	}

	void dispatch_event(const SDL_Event& event) {
		extern std::atomic<bool> running;
		switch (event.type) {
			case SDL_QUIT:
				running.store(false, std::memory_order_relaxed);
				break;
			case SDL_WINDOWEVENT:
				paint();
				break;
			case SDL_MOUSEMOTION:
				onMouseMove(event);
				break;
			case SDL_MOUSEBUTTONUP:
				onMouseButtonUp(event);
				break;
			case SDL_MOUSEBUTTONDOWN:
				onMouseButtonDown(event);
				break;
			default:
				break;
		}
	}

	void event_loop()
	{
		SDL_Event event;
		extern std::atomic<bool> running;
		while (running.load(std::memory_order_relaxed)) {
			if (SDL_WaitEventTimeout(&event, 16)) {
				do {
					dispatch_event(event);
				} while(SDL_PollEvent(&event));
			}

			paint();
		}
	}

	void paint()
	{
		auto renderer = this->renderer.get();
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		SDL_RenderClear(renderer);
		SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255);
		SDL_RenderDrawRect(renderer, &boundary_rect);
		SDL_Rect rect;
		rect.w = pixels_per_cell;
		rect.h = pixels_per_cell;

		{
			std::deque<ScalarField> field_queue = dye_field_to_ui->try_pop_all();
			ScalarField field;
			if (not field_queue.empty()) {
				if (not this->field.empty()) {
					std::swap(this->field, field_queue.back());
				}
				if (not field.empty()) {
					this->field = std::move(field);
				}

			}
		}

		for (uint x = 1; x < cells - 2; ++x) {
			for (uint y = 1; y < cells - 2; ++y) {
				rect.x = x * pixels_per_cell;
				rect.y = y * pixels_per_cell;
				auto field_val = field[y * cells + x];
				if (field_val < 0.0) {
					SDL_SetRenderDrawColor(renderer, 0, std::min(fabs(255 * field[y * cells + x]), 255.0), 0, 255);
				} else {
					SDL_SetRenderDrawColor(renderer, std::min(255 * field[y * cells + x], 255.0f), 0, 0, 255);
				}

				SDL_RenderFillRect(renderer, &rect);
			}
		}

		SDL_RenderPresent(renderer);
	}
};
#endif //MAINWINDOW_H
