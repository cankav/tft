/* 
 * File:   randomcache.hpp
 * Author: Can Kavaklioglu
 * Derived from: randomcache.hpp of Alexander Ponomarev
 *
 */

#ifndef _RANDOMCACHE_HPP_INCLUDED_
#define	_RANDOMCACHE_HPP_INCLUDED_

#include <unordered_map>
#include <list>
#include <cstddef>
#include <stdexcept>

namespace random_cache {

template<typename key_t, typename value_t>
class random_cache {
  public:
    typedef typename std::pair<key_t, value_t> key_value_pair_t;

    random_cache(size_t max_size) :
      _max_size(max_size){
    }

    // default constructor
    random_cache(){}

    void put(const key_t& key, const value_t& value) {
      if (_cache_items_map.size() == _max_size-1) {
        auto it = _cache_items_map.cbegin();
        _cache_items_map.erase(it);
      }
      _cache_items_map[key] = value;
    }

    const value_t& get(const key_t& key) {
      return _cache_items_map.at(key);
      // auto it = _cache_items_map.find(key);
      // if (it == _cache_items_map.end()) {
      //   throw std::range_error("There is no such key in cache");
      // } else {
      //   return it->second;
      // }
    }
	
    bool exists(const key_t& key) const {
      return _cache_items_map.find(key) != _cache_items_map.end();
    }
	
    size_t size() const {
      return _cache_items_map.size();
    }
	
  private:
    std::unordered_map<key_t, value_t> _cache_items_map;
    size_t _max_size;
};

} // namespace random_cache

#endif	/* _RANDOMCACHE_HPP_INCLUDED_ */

