// :bustub-keep-private:
//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// arc_replacer.cpp
//
// Identification: src/buffer/arc_replacer.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "buffer/arc_replacer.h"
#include <algorithm>
#include <optional>
#include <stdexcept>
#include "common/config.h"

namespace bustub {

/**
 * @brief a new ArcReplacer, with lists initialized to be empty and target size to 0
 * @param num_frames the maximum number of frames the ArcReplacer will be required to cache
 */
ArcReplacer::ArcReplacer(size_t num_frames) : replacer_size_(num_frames) {}

/**
 * Evict the LRU-most evictable frame from one side (MRU or MFU) and move its page
 * into the corresponding ghost list. Caller holds latch_.
 */
auto ArcReplacer::EvictFrom(bool from_mru) -> std::optional<frame_id_t> {
  auto &lst = from_mru ? mru_ : mfu_;
  auto &ghost = from_mru ? mru_ghost_ : mfu_ghost_;
  auto ghost_status = from_mru ? ArcStatus::MRU_GHOST : ArcStatus::MFU_GHOST;

  // Scan from the back (least recently used) for an evictable frame.
  for (auto it = lst.rbegin(); it != lst.rend(); ++it) {
    frame_id_t fid = *it;
    auto status = alive_map_[fid];
    if (!status->evictable_) {
      continue;
    }
    page_id_t pid = status->page_id_;

    lst.erase(alive_iter_[fid]);
    alive_iter_.erase(fid);
    alive_map_.erase(fid);

    ghost.push_front(pid);
    ghost_iter_[pid] = ghost.begin();
    status->arc_status_ = ghost_status;
    status->evictable_ = false;
    ghost_map_[pid] = status;

    curr_size_--;
    return fid;
  }
  return std::nullopt;
}

/**
 * @brief Performs the Replace operation as described by the writeup
 * that evicts from either mfu_ or mru_ into its corresponding ghost list
 * according to balancing policy.
 *
 * @return frame id of the evicted frame, or std::nullopt if cannot evict
 */
auto ArcReplacer::Evict() -> std::optional<frame_id_t> {
  std::scoped_lock<std::mutex> guard(latch_);
  if (curr_size_ == 0) {
    return std::nullopt;
  }

  // MRU smaller than target -> prefer MFU; otherwise prefer MRU. Fall back to the other side.
  bool prefer_mru = mru_.size() >= mru_target_size_;
  auto result = EvictFrom(prefer_mru);
  if (!result.has_value()) {
    result = EvictFrom(!prefer_mru);
  }
  return result;
}

void ArcReplacer::DropGhostTail(bool from_mru_ghost) {
  auto &ghost = from_mru_ghost ? mru_ghost_ : mfu_ghost_;
  if (ghost.empty()) {
    return;
  }
  page_id_t pid = ghost.back();
  ghost.pop_back();
  ghost_iter_.erase(pid);
  ghost_map_.erase(pid);
}

/**
 * @brief Record access to a frame, adjusting ARC bookkeeping accordingly.
 *
 * @param frame_id id of frame that received a new access.
 * @param page_id id of page that is mapped to the frame.
 * @param access_type type of access that was received (unused).
 */
void ArcReplacer::RecordAccess(frame_id_t frame_id, page_id_t page_id, [[maybe_unused]] AccessType access_type) {
  std::scoped_lock<std::mutex> guard(latch_);

  // Case 1: cache hit in MRU or MFU.
  auto alive_it = alive_map_.find(frame_id);
  if (alive_it != alive_map_.end()) {
    auto &status = alive_it->second;
    auto &pos = alive_iter_[frame_id];
    if (status->arc_status_ == ArcStatus::MRU) {
      mfu_.splice(mfu_.begin(), mru_, pos);
      status->arc_status_ = ArcStatus::MFU;
    } else {
      mfu_.splice(mfu_.begin(), mfu_, pos);
    }
    pos = mfu_.begin();
    return;
  }

  auto ghost_it = ghost_map_.find(page_id);
  if (ghost_it != ghost_map_.end()) {
    auto status = ghost_it->second;
    bool in_mru_ghost = status->arc_status_ == ArcStatus::MRU_GHOST;
    size_t mru_g = mru_ghost_.size();
    size_t mfu_g = mfu_ghost_.size();

    if (in_mru_ghost) {
      // Case 2: ghost hit in MRU ghost -> grow MRU target.
      size_t delta = (mru_g >= mfu_g) ? 1 : mfu_g / mru_g;
      mru_target_size_ = std::min(mru_target_size_ + delta, replacer_size_);
      mru_ghost_.erase(ghost_iter_[page_id]);
    } else {
      // Case 3: ghost hit in MFU ghost -> shrink MRU target.
      size_t delta = (mfu_g >= mru_g) ? 1 : mru_g / mfu_g;
      mru_target_size_ = (delta >= mru_target_size_) ? 0 : mru_target_size_ - delta;
      mfu_ghost_.erase(ghost_iter_[page_id]);
    }
    ghost_iter_.erase(page_id);
    ghost_map_.erase(page_id);

    // Revive into the front of MFU; stays non-evictable until the caller says otherwise.
    status->frame_id_ = frame_id;
    status->page_id_ = page_id;
    status->evictable_ = false;
    status->arc_status_ = ArcStatus::MFU;
    mfu_.push_front(frame_id);
    alive_iter_[frame_id] = mfu_.begin();
    alive_map_[frame_id] = status;
    return;
  }

  // Case 4: complete miss.
  if (mru_.size() + mru_ghost_.size() >= replacer_size_) {
    DropGhostTail(true);
  } else if (mru_.size() + mru_ghost_.size() + mfu_.size() + mfu_ghost_.size() >= 2 * replacer_size_) {
    DropGhostTail(false);
  }
  mru_.push_front(frame_id);
  alive_iter_[frame_id] = mru_.begin();
  alive_map_[frame_id] = std::make_shared<FrameStatus>(page_id, frame_id, false, ArcStatus::MRU);
}

/**
 * @brief Toggle whether a frame is evictable or non-evictable.
 *
 * @param frame_id id of frame whose 'evictable' status will be modified
 * @param set_evictable whether the given frame is evictable or not
 */
void ArcReplacer::SetEvictable(frame_id_t frame_id, bool set_evictable) {
  std::scoped_lock<std::mutex> guard(latch_);
  if (frame_id < 0 || static_cast<size_t>(frame_id) > replacer_size_) {
    throw std::invalid_argument("ArcReplacer::SetEvictable: invalid frame id");
  }
  auto it = alive_map_.find(frame_id);
  if (it == alive_map_.end()) {
    return;
  }
  auto &status = it->second;
  if (status->evictable_ == set_evictable) {
    return;
  }
  status->evictable_ = set_evictable;
  if (set_evictable) {
    curr_size_++;
  } else {
    curr_size_--;
  }
}

/**
 * @brief Remove an evictable frame from replacer (no ghost entry is created).
 *
 * @param frame_id id of frame to be removed
 */
void ArcReplacer::Remove(frame_id_t frame_id) {
  std::scoped_lock<std::mutex> guard(latch_);
  auto it = alive_map_.find(frame_id);
  if (it == alive_map_.end()) {
    return;
  }
  if (!it->second->evictable_) {
    throw std::logic_error("ArcReplacer::Remove: frame is not evictable");
  }
  if (it->second->arc_status_ == ArcStatus::MRU) {
    mru_.erase(alive_iter_[frame_id]);
  } else {
    mfu_.erase(alive_iter_[frame_id]);
  }
  alive_iter_.erase(frame_id);
  alive_map_.erase(it);
  curr_size_--;
}

/**
 * @brief Return replacer's size, which tracks the number of evictable frames.
 *
 * @return size_t
 */
auto ArcReplacer::Size() -> size_t {
  std::scoped_lock<std::mutex> guard(latch_);
  return curr_size_;
}

}  // namespace bustub
