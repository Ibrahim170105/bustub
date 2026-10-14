//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// page_guard.cpp
//
// Identification: src/storage/page/page_guard.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "storage/page/page_guard.h"
#include <memory>
#include <utility>
#include <vector>
#include "buffer/arc_replacer.h"
#include "common/macros.h"

namespace bustub {

namespace {

/** Synchronously writes `data` to `page_id` through the disk scheduler. */
void WriteThrough(DiskScheduler *scheduler, page_id_t page_id, const char *data) {
  auto promise = scheduler->CreatePromise();
  auto future = promise.get_future();
  std::vector<DiskRequest> requests;
  requests.push_back(DiskRequest{true, const_cast<char *>(data), page_id, std::move(promise)});
  scheduler->Schedule(requests);
  future.get();
}

}  // namespace

/**
 * @brief The only constructor for an RAII `ReadPageGuard` that creates a valid guard.
 *
 * The shared page latch is acquired here. The caller (the buffer pool manager) has already pinned the frame and
 * released the BPM latch, so blocking here cannot deadlock with the BPM.
 */
ReadPageGuard::ReadPageGuard(page_id_t page_id, std::shared_ptr<FrameHeader> frame,
                             std::shared_ptr<ArcReplacer> replacer, std::shared_ptr<std::mutex> bpm_latch,
                             std::shared_ptr<DiskScheduler> disk_scheduler)
    : page_id_(page_id),
      frame_(std::move(frame)),
      replacer_(std::move(replacer)),
      bpm_latch_(std::move(bpm_latch)),
      disk_scheduler_(std::move(disk_scheduler)) {
  frame_->rwlatch_.lock_shared();
  is_valid_ = true;
}

/**
 * @brief The move constructor for `ReadPageGuard`. The other guard is invalidated.
 */
ReadPageGuard::ReadPageGuard(ReadPageGuard &&that) noexcept
    : page_id_(that.is_valid_ ? that.page_id_ : INVALID_PAGE_ID),
      frame_(std::move(that.frame_)),
      replacer_(std::move(that.replacer_)),
      bpm_latch_(std::move(that.bpm_latch_)),
      disk_scheduler_(std::move(that.disk_scheduler_)),
      is_valid_(that.is_valid_) {
  that.is_valid_ = false;
}

/**
 * @brief The move assignment operator for `ReadPageGuard`. Releases whatever this guard currently holds first.
 */
auto ReadPageGuard::operator=(ReadPageGuard &&that) noexcept -> ReadPageGuard & {
  if (this != &that) {
    Drop();
    page_id_ = that.is_valid_ ? that.page_id_ : INVALID_PAGE_ID;
    frame_ = std::move(that.frame_);
    replacer_ = std::move(that.replacer_);
    bpm_latch_ = std::move(that.bpm_latch_);
    disk_scheduler_ = std::move(that.disk_scheduler_);
    is_valid_ = that.is_valid_;
    that.is_valid_ = false;
  }
  return *this;
}

/**
 * @brief Gets the page ID of the page this guard is protecting.
 */
auto ReadPageGuard::GetPageId() const -> page_id_t {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid read guard");
  return page_id_;
}

/**
 * @brief Gets a `const` pointer to the page of data this guard is protecting.
 */
auto ReadPageGuard::GetData() const -> const char * {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid read guard");
  return frame_->GetData();
}

/**
 * @brief Returns whether the page is dirty (modified but not flushed to the disk).
 */
auto ReadPageGuard::IsDirty() const -> bool {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid read guard");
  return frame_->is_dirty_;
}

/**
 * @brief Flushes this page's data safely to disk. We already hold the page latch; the BPM latch serializes access to
 * the dirty flag.
 */
void ReadPageGuard::Flush() {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid read guard");
  std::scoped_lock latch(*bpm_latch_);
  if (frame_->is_dirty_) {
    WriteThrough(disk_scheduler_.get(), page_id_, frame_->GetData());
    frame_->is_dirty_ = false;
  }
}

/**
 * @brief Manually drops a valid `ReadPageGuard`'s data. If this guard is invalid, this function does nothing.
 *
 * Order matters: release the page latch first, then take the BPM latch to unpin and update the replacer. We never hold
 * the page latch while waiting for the BPM latch in a way the BPM could be waiting on in reverse (the BPM never waits
 * for a page latch while holding its own latch).
 */
void ReadPageGuard::Drop() {
  if (!is_valid_) {
    return;
  }
  frame_->rwlatch_.unlock_shared();
  {
    std::scoped_lock latch(*bpm_latch_);
    if (frame_->pin_count_.fetch_sub(1) == 1) {
      replacer_->SetEvictable(frame_->frame_id_, true);
    }
  }
  is_valid_ = false;
  frame_.reset();
  replacer_.reset();
  bpm_latch_.reset();
  disk_scheduler_.reset();
}

/** @brief The destructor for `ReadPageGuard`. This destructor simply calls `Drop()`. */
ReadPageGuard::~ReadPageGuard() { Drop(); }

/**********************************************************************************************************************/
/**********************************************************************************************************************/
/**********************************************************************************************************************/

/**
 * @brief The only constructor for an RAII `WritePageGuard` that creates a valid guard.
 *
 * The exclusive page latch is acquired here (see `ReadPageGuard`'s constructor for why this is safe).
 */
WritePageGuard::WritePageGuard(page_id_t page_id, std::shared_ptr<FrameHeader> frame,
                               std::shared_ptr<ArcReplacer> replacer, std::shared_ptr<std::mutex> bpm_latch,
                               std::shared_ptr<DiskScheduler> disk_scheduler)
    : page_id_(page_id),
      frame_(std::move(frame)),
      replacer_(std::move(replacer)),
      bpm_latch_(std::move(bpm_latch)),
      disk_scheduler_(std::move(disk_scheduler)) {
  frame_->rwlatch_.lock();
  is_valid_ = true;
}

/**
 * @brief The move constructor for `WritePageGuard`. The other guard is invalidated.
 */
WritePageGuard::WritePageGuard(WritePageGuard &&that) noexcept
    : page_id_(that.is_valid_ ? that.page_id_ : INVALID_PAGE_ID),
      frame_(std::move(that.frame_)),
      replacer_(std::move(that.replacer_)),
      bpm_latch_(std::move(that.bpm_latch_)),
      disk_scheduler_(std::move(that.disk_scheduler_)),
      is_valid_(that.is_valid_) {
  that.is_valid_ = false;
}

/**
 * @brief The move assignment operator for `WritePageGuard`. Releases whatever this guard currently holds first.
 */
auto WritePageGuard::operator=(WritePageGuard &&that) noexcept -> WritePageGuard & {
  if (this != &that) {
    Drop();
    page_id_ = that.is_valid_ ? that.page_id_ : INVALID_PAGE_ID;
    frame_ = std::move(that.frame_);
    replacer_ = std::move(that.replacer_);
    bpm_latch_ = std::move(that.bpm_latch_);
    disk_scheduler_ = std::move(that.disk_scheduler_);
    is_valid_ = that.is_valid_;
    that.is_valid_ = false;
  }
  return *this;
}

/**
 * @brief Gets the page ID of the page this guard is protecting.
 */
auto WritePageGuard::GetPageId() const -> page_id_t {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid write guard");
  return page_id_;
}

/**
 * @brief Gets a `const` pointer to the page of data this guard is protecting.
 */
auto WritePageGuard::GetData() const -> const char * {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid write guard");
  return frame_->GetData();
}

/**
 * @brief Gets a mutable pointer to the page of data this guard is protecting.
 */
auto WritePageGuard::GetDataMut() -> char * {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid write guard");
  return frame_->GetDataMut();
}

/**
 * @brief Returns whether the page is dirty (modified but not flushed to the disk).
 */
auto WritePageGuard::IsDirty() const -> bool {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid write guard");
  return frame_->is_dirty_;
}

/**
 * @brief Flushes this page's data safely to disk. We already hold the exclusive page latch.
 */
void WritePageGuard::Flush() {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid write guard");
  std::scoped_lock latch(*bpm_latch_);
  if (frame_->is_dirty_) {
    WriteThrough(disk_scheduler_.get(), page_id_, frame_->GetData());
    frame_->is_dirty_ = false;
  }
}

/**
 * @brief Manually drops a valid `WritePageGuard`'s data. If this guard is invalid, this function does nothing.
 *
 * The page is conservatively marked dirty on drop (under the BPM latch), so that modifications made through this guard
 * after a `Flush()` are never lost on eviction.
 */
void WritePageGuard::Drop() {
  if (!is_valid_) {
    return;
  }
  frame_->rwlatch_.unlock();
  {
    std::scoped_lock latch(*bpm_latch_);
    frame_->is_dirty_ = true;
    if (frame_->pin_count_.fetch_sub(1) == 1) {
      replacer_->SetEvictable(frame_->frame_id_, true);
    }
  }
  is_valid_ = false;
  frame_.reset();
  replacer_.reset();
  bpm_latch_.reset();
  disk_scheduler_.reset();
}

/** @brief The destructor for `WritePageGuard`. This destructor simply calls `Drop()`. */
WritePageGuard::~WritePageGuard() { Drop(); }

}  // namespace bustub
