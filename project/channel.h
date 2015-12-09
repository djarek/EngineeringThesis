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

#ifndef CHANNEL_H
#define CHANNEL_H

#include <memory>
#include <mutex>
#include <deque>

template<typename T>
class Channel
{
	std::mutex mutex;
	std::deque<T> items;
public:
	static std::shared_ptr<Channel> make()
	{
		return std::make_shared<Channel>();
	}

	bool try_push(T& item)
	{
		std::unique_lock<std::mutex> guard{mutex, std::try_to_lock_t{}};
		if (not guard.owns_lock()) {
			return false;
		}
		items.emplace_back(std::move(item));
		return true;
	}

	bool try_pop(T& item)
	{
		std::unique_lock<std::mutex> guard{mutex, std::try_to_lock_t{}};
		if (not guard.owns_lock()) {
			return false;
		}
		if (not items.empty()) {
			item = std::move(items.front());
			items.pop_front();
			return true;
		} else {
			return false;
		}
	}

	std::deque<T> try_pop_all()
	{
		std::unique_lock<std::mutex> guard{mutex, std::try_to_lock_t{}};
		std::deque<T> ret;
		if (guard.owns_lock()) {
			ret = std::move(items);
		}
		return ret;
	}

	bool try_push_all(std::deque<T>& items)
	{
		std::unique_lock<std::mutex> guard{mutex, std::try_to_lock_t{}};
		if (guard.owns_lock()) {
			std::move(items.begin(), items.end(), std::back_inserter(this->items));
			items.clear();
			return true;
		} else {
			return false;
		}

	}
};

template<typename T>
using Channel_ptr = std::shared_ptr<Channel<T>>;

#endif //CHANNEL_H
