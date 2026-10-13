#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "common/util/hash_util.h"

namespace bustub {

/** Avalanches a deterministic key value so power-of-two capacities use entropy from every hash bit. */
inline auto MixRobinHoodHash(uint64_t hash) -> size_t {
  hash ^= hash >> 33U;
  hash *= 0xff51afd7ed558ccdULL;
  hash ^= hash >> 33U;
  hash *= 0xc4ceb9fe1a85ec53ULL;
  return hash ^ (hash >> 33U);
}

/** Deterministic key hash for supported types; strings hash their bytes through HashUtil before mixing. */
template <typename KeyType>
struct RobinHoodHash;

template <>
struct RobinHoodHash<int> {
  auto operator()(int key) const -> size_t {
    return static_cast<size_t>(static_cast<uint64_t>(static_cast<int64_t>(key)) * 0x9e3779b97f4a7c15ULL);
  }
};

template <>
struct RobinHoodHash<int64_t> {
  auto operator()(int64_t key) const -> size_t {
    return static_cast<size_t>(static_cast<uint64_t>(key) * 0x9e3779b97f4a7c15ULL);
  }
};

template <>
struct RobinHoodHash<std::string> {
  auto operator()(const std::string &key) const -> size_t {
    return MixRobinHoodHash(HashUtil::HashBytes(key.data(), key.size()));
  }
};

/** A fixed-capacity, concurrent hash set using Robin Hood open addressing. */
template <typename KeyType, typename Hash = RobinHoodHash<KeyType>, typename KeyEqual = std::equal_to<KeyType>>
class RobinHoodHashSet {
 public:
  explicit RobinHoodHashSet(size_t capacity, Hash hasher = Hash{}, KeyEqual key_equal = KeyEqual{});

  RobinHoodHashSet() = delete;
  RobinHoodHashSet(const RobinHoodHashSet &) = delete;
  auto operator=(const RobinHoodHashSet &) -> RobinHoodHashSet & = delete;

  RobinHoodHashSet(RobinHoodHashSet &&other) noexcept;
  auto operator=(RobinHoodHashSet &&other) noexcept -> RobinHoodHashSet &;

  /** @return true when a key is inserted or an existing equal key is replaced, false when full. */
  auto Insert(const KeyType &key) -> bool;
  /** @return true if key was present and was marked deleted. */
  auto Remove(const KeyType &key) -> bool;
  void Clear();

  auto Contains(const KeyType &key) const -> bool;
  auto GetBucket(const KeyType &key) const -> size_t;
  auto Size() const -> size_t;
  auto Capacity() const -> size_t;
  auto BucketCount() const -> size_t { return Capacity(); }
  auto LoadFactor() const -> double;
  auto MaxProbeDistance() const -> size_t;

 private:
  /** The three logical states a bucket can be in. */
  enum class BucketState : uint8_t { kEmpty = 0, kOccupied = 1, kTombstone = 2 };

  /** A single slot of the fixed-capacity table. */
  struct Bucket {
    KeyType key_{};
    size_t probe_distance_{0};
    BucketState state_{BucketState::kEmpty};
  };

  /**
   * A latch guarding one contiguous block of buckets, plus that block's live-entry count.
   * alignas(64) keeps neighbouring stripes off the same cache line (avoids false sharing).
   */
  struct alignas(64) Stripe {
    std::mutex latch_;
    std::atomic<size_t> size_{0};
  };

  /**
   * Holds the stripe latches an operation has taken, and releases them on scope exit.
   *
   * Invariant: latches are only ever acquired in increasing stripe order, so the held set is
   * the contiguous range [first_, first_ + count_). Acquire() reports failure instead of going
   * backwards, which keeps the global lock order intact and makes deadlock impossible.
   */
  class StripeGuard {
   public:
    StripeGuard(Stripe *stripes, size_t stripe_count) : stripes_(stripes), stripe_count_(stripe_count) {}
    ~StripeGuard() { Release(); }

    StripeGuard(const StripeGuard &) = delete;
    auto operator=(const StripeGuard &) -> StripeGuard & = delete;
    StripeGuard(StripeGuard &&) = delete;
    auto operator=(StripeGuard &&) -> StripeGuard & = delete;

    /** @return false when the probe wrapped and the caller must fall back to LockAll(). */
    auto Acquire(size_t stripe) -> bool {
      if (count_ == 0) {
        stripes_[stripe].latch_.lock();
        first_ = stripe;
        count_ = 1;
        return true;
      }
      if (stripe < first_) {
        return false;  // The probe wrapped; taking this latch now would violate the lock order.
      }
      const size_t last = first_ + count_ - 1;
      for (size_t next = last + 1; next <= stripe; next++) {
        stripes_[next].latch_.lock();
        count_++;
      }
      return true;
    }

    /** Whole-table path: still ascending order, and never needs another latch afterwards. */
    void LockAll() {
      Release();
      for (size_t stripe = 0; stripe < stripe_count_; stripe++) {
        stripes_[stripe].latch_.lock();
      }
      first_ = 0;
      count_ = stripe_count_;
    }

    void Release() {
      for (size_t i = 0; i < count_; i++) {
        stripes_[first_ + i].latch_.unlock();
      }
      count_ = 0;
    }

   private:
    Stripe *stripes_;
    size_t stripe_count_;
    size_t first_{0};
    size_t count_{0};
  };

  /** Upper bound on stripes; more than this wastes memory without reducing contention further. */
  static constexpr size_t kMaxStripes = 1024;

  auto HomeBucket(const KeyType &key) const -> size_t;
  auto NextBucket(size_t bucket) const -> size_t;
  auto StripeOf(size_t bucket) const -> size_t;

  /** Each returns false only when the probe wrapped, leaving the table unmodified. */
  auto RunInsert(const KeyType &key, StripeGuard *guard, bool *inserted) -> bool;
  auto RunRemove(const KeyType &key, StripeGuard *guard, bool *removed) -> bool;
  auto RunFind(const KeyType &key, StripeGuard *guard, size_t *bucket) const -> bool;

  std::vector<Bucket> buckets_;
  std::unique_ptr<Stripe[]> stripes_;
  size_t capacity_{0};
  size_t stripe_count_{0};
  size_t buckets_per_stripe_{0};
  Hash hasher_;
  KeyEqual key_equal_;
};

}  // namespace bustub