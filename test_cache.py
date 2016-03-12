from cache import StorageCache

file = 'test_cache.json'
cache = StorageCache(file)

data = cache.get()
data['b'] = 10
cache.save()

cache2 = StorageCache(file)

assert cache2.has('b')