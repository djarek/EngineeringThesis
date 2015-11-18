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
	static std::shared_ptr<Channel> make() {
		return std::make_shared<Channel>();
	}

	bool try_push(T& item) {
		std::lock_guard<std::mutex> guard{mutex};
		items.push_back(std::move(item));
		return true;
	}

	bool try_pop(T& item) {
		std::lock_guard<std::mutex> guard{mutex};
		if (not items.empty()) {
			item = std::move(items.front());
			items.pop_front();
			return true;
		} else {
			return false;
		}
	}
};

template<typename T>
using Channel_ptr = std::shared_ptr<Channel<T>>;

#endif //CHANNEL_H
