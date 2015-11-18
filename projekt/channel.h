#ifndef CHANNEL_H
#define CHANNEL_H

#include <memory>
#include <mutex>

template<typename T>
class Channel
{
	std::mutex mutex;
	T item;
	bool empty{true};
public:
	static std::shared_ptr<Channel> make() {
		return std::make_shared<Channel>();
	}

	bool try_push(T& item) {
		std::lock_guard<std::mutex> guard{mutex};
		if (empty) {
			this->item = std::move(item);
			empty = false;
			return true;
		} else {
			return false;
		}
	}

	bool try_pop(T& item) {
		std::lock_guard<std::mutex> guard{mutex};
		if (not empty) {
			item = std::move(this->item);
			empty = true;
			return true;
		} else {
			return false;
		}
	}
};

template<typename T>
using Channel_ptr = std::shared_ptr<Channel<T>>;

#endif //CHANNEL_H
