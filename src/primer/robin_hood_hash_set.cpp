//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// robin_hood_hash_set.cpp
//
// Identification: src/primer/robin_hood_hash_set.cpp
//
//===----------------------------------------------------------------------===//

#include "primer/robin_hood_hash_set.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace bustub {

template <typename KeyType, typename Hash, typename KeyEqual>
RobinHoodHashSet<KeyType, Hash, KeyEqual>::RobinHoodHashSet(size_t capacity, Hash hasher, KeyEqual key_equal)
    : hasher_(std::move(hasher)), key_equal_(std::move(key_equal)) {
  if (capacity == 0) {
    throw std::invalid_argument("RobinHoodHashSet requires a nonzero capacity");
  }
  capacity_ = capacity;
  buckets_.assign(capacity_, Bucket{});

  // Consecutive buckets share a stripe, so a short probe usually takes a single latch.
  buckets_per_stripe_ = std::max<size_t>(1, (capacity_ + kMaxStripes - 1) / kMaxStripes);
  stripe_count_ = (capacity_ + buckets_per_stripe_ - 1) / buckets_per_stripe_;
  stripes_ = std::make_unique<Stripe[]>(stripe_count_);
}

template <typename KeyType, typename Hash, typename KeyEqual>
RobinHoodHashSet<KeyType, Hash, KeyEqual>::RobinHoodHashSet(RobinHoodHashSet &&other) noexcept
    : buckets_(std::move(other.buckets_)),
      stripes_(std::move(other.stripes_)),
      capacity_(other.capacity_),
      stripe_count_(other.stripe_count_),
      buckets_per_stripe_(other.buckets_per_stripe_),
      hasher_(std::move(other.hasher_)),
      key_equal_(std::move(other.key_equal_)) {
  other.buckets_.clear();
  other.capacity_ = 0;
  other.stripe_count_ = 0;
  other.buckets_per_stripe_ = 0;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::operator=(RobinHoodHashSet &&other) noexcept -> RobinHoodHashSet & {
  if (this == &other) {
    return *this;
  }
  buckets_ = std::move(other.buckets_);
  stripes_ = std::move(other.stripes_);
  capacity_ = other.capacity_;
  stripe_count_ = other.stripe_count_;
  buckets_per_stripe_ = other.buckets_per_stripe_;
  hasher_ = std::move(other.hasher_);
  key_equal_ = std::move(other.key_equal_);

  other.buckets_.clear();
  other.capacity_ = 0;
  other.stripe_count_ = 0;
  other.buckets_per_stripe_ = 0;
  return *this;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::HomeBucket(const KeyType &key) const -> size_t {
  return hasher_(key) % capacity_;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::NextBucket(size_t bucket) const -> size_t {
  const size_t next = bucket + 1;
  return next == capacity_ ? 0 : next;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::StripeOf(size_t bucket) const -> size_t {
  return bucket / buckets_per_stripe_;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::RunInsert(const KeyType &key, StripeGuard *guard, bool *inserted)
    -> bool {
  const size_t home = HomeBucket(key);

  // Phase 1: scan for a duplicate, for the slot we may claim, and for a terminator.
  size_t insert_pos = capacity_;
  size_t insert_distance = 0;
  bool have_terminator = false;
  size_t bucket = home;

  for (size_t distance = 0; distance < capacity_; distance++) {
    if (!guard->Acquire(StripeOf(bucket))) {
      return false;  // Wrapped; nothing has been modified yet.
    }
    const Bucket &slot = buckets_[bucket];

    if (slot.state_ == BucketState::kEmpty) {
      if (insert_pos == capacity_) {
        insert_pos = bucket;
        insert_distance = distance;
      }
      have_terminator = true;
      break;  // The key cannot live past an empty bucket.
    }

    if (slot.state_ == BucketState::kTombstone) {
      if (insert_pos == capacity_) {
        insert_pos = bucket;
        insert_distance = distance;
      }
      have_terminator = true;
      // Keep scanning: an equal key may still live further down the chain.
    } else {
      if (key_equal_(slot.key_, key)) {
        *inserted = true;  // Duplicate: a successful insert that does not change Size().
        return true;
      }
      if (insert_pos == capacity_ && slot.probe_distance_ < distance) {
        insert_pos = bucket;  // Robin Hood: we are poorer than the resident, so we take this slot.
        insert_distance = distance;
      }
    }
    bucket = NextBucket(bucket);
  }

  if (!have_terminator || insert_pos == capacity_) {
    *inserted = false;  // Every bucket is occupied: the cascade would never terminate.
    return true;
  }

  // Phase 2: cascade. It cannot run past the terminator found above, so every bucket it
  // touches was already visited (and latched) in phase 1.
  KeyType carry_key = key;
  size_t carry_distance = insert_distance;
  size_t index = insert_pos;
  while (true) {
    Bucket &slot = buckets_[index];
    if (slot.state_ != BucketState::kOccupied) {
      slot.key_ = std::move(carry_key);
      slot.probe_distance_ = carry_distance;
      slot.state_ = BucketState::kOccupied;
      break;
    }
    if (slot.probe_distance_ < carry_distance) {
      // Swap the distance too: the evicted key continues from its own distance, not ours.
      std::swap(slot.key_, carry_key);
      std::swap(slot.probe_distance_, carry_distance);
    }
    index = NextBucket(index);
    carry_distance++;
  }

  stripes_[StripeOf(home)].size_.fetch_add(1, std::memory_order_relaxed);
  *inserted = true;
  return true;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::RunRemove(const KeyType &key, StripeGuard *guard, bool *removed)
    -> bool {
  const size_t home = HomeBucket(key);
  size_t bucket = home;

  for (size_t distance = 0; distance < capacity_; distance++) {
    if (!guard->Acquire(StripeOf(bucket))) {
      return false;
    }
    Bucket &slot = buckets_[bucket];

    if (slot.state_ == BucketState::kEmpty) {
      break;
    }
    if (slot.state_ == BucketState::kOccupied && key_equal_(slot.key_, key)) {
      slot.state_ = BucketState::kTombstone;
      slot.key_ = KeyType{};  // Release anything the key owns; tombstone keys are never read.
      slot.probe_distance_ = 0;
      stripes_[StripeOf(home)].size_.fetch_sub(1, std::memory_order_relaxed);
      *removed = true;
      return true;
    }
    bucket = NextBucket(bucket);
  }

  *removed = false;
  return true;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::RunFind(const KeyType &key, StripeGuard *guard, size_t *bucket) const
    -> bool {
  size_t index = HomeBucket(key);

  for (size_t distance = 0; distance < capacity_; distance++) {
    if (!guard->Acquire(StripeOf(index))) {
      return false;
    }
    const Bucket &slot = buckets_[index];

    if (slot.state_ == BucketState::kEmpty) {
      break;  // Only an empty bucket may terminate a lookup.
    }
    if (slot.state_ == BucketState::kOccupied && key_equal_(slot.key_, key)) {
      *bucket = index;
      return true;
    }
    index = NextBucket(index);  // Tombstones and non-matching keys are probed past.
  }

  *bucket = capacity_;
  return true;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::Insert(const KeyType &key) -> bool {
  if (capacity_ == 0) {
    return false;
  }
  StripeGuard guard(stripes_.get(), stripe_count_);
  bool inserted = false;
  if (!RunInsert(key, &guard, &inserted)) {
    guard.LockAll();  // Holds every stripe, so the retry cannot wrap out of order again.
    RunInsert(key, &guard, &inserted);
  }
  return inserted;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::Remove(const KeyType &key) -> bool {
  if (capacity_ == 0) {
    return false;
  }
  StripeGuard guard(stripes_.get(), stripe_count_);
  bool removed = false;
  if (!RunRemove(key, &guard, &removed)) {
    guard.LockAll();
    RunRemove(key, &guard, &removed);
  }
  return removed;
}

template <typename KeyType, typename Hash, typename KeyEqual>
void RobinHoodHashSet<KeyType, Hash, KeyEqual>::Clear() {
  if (capacity_ == 0) {
    return;  // A moved-from table has nothing to clear but must stay usable.
  }
  StripeGuard guard(stripes_.get(), stripe_count_);
  guard.LockAll();

  for (auto &slot : buckets_) {
    slot = Bucket{};  // Resets the state to kEmpty, wiping tombstones too.
  }
  for (size_t stripe = 0; stripe < stripe_count_; stripe++) {
    stripes_[stripe].size_.store(0, std::memory_order_relaxed);
  }
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::Contains(const KeyType &key) const -> bool {
  return GetBucket(key) != capacity_;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::GetBucket(const KeyType &key) const -> size_t {
  if (capacity_ == 0) {
    return capacity_;
  }
  StripeGuard guard(stripes_.get(), stripe_count_);
  size_t bucket = capacity_;
  if (!RunFind(key, &guard, &bucket)) {
    guard.LockAll();
    RunFind(key, &guard, &bucket);
  }
  return bucket;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::Size() const -> size_t {
  size_t total = 0;
  for (size_t stripe = 0; stripe < stripe_count_; stripe++) {
    total += stripes_[stripe].size_.load(std::memory_order_relaxed);
  }
  return total;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::Capacity() const -> size_t {
  return capacity_;
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::LoadFactor() const -> double {
  if (capacity_ == 0) {
    return 0.0;
  }
  return static_cast<double>(Size()) / static_cast<double>(capacity_);
}

template <typename KeyType, typename Hash, typename KeyEqual>
auto RobinHoodHashSet<KeyType, Hash, KeyEqual>::MaxProbeDistance() const -> size_t {
  if (capacity_ == 0) {
    return 0;
  }
  StripeGuard guard(stripes_.get(), stripe_count_);
  guard.LockAll();

  size_t max_distance = 0;
  for (const auto &slot : buckets_) {
    if (slot.state_ == BucketState::kOccupied) {
      max_distance = std::max(max_distance, slot.probe_distance_);
    }
  }
  return max_distance;
}

template class RobinHoodHashSet<int>;
template class RobinHoodHashSet<int64_t>;
template class RobinHoodHashSet<std::string>;

}  // namespace bustub