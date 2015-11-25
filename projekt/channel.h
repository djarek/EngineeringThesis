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
