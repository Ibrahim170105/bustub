//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// buffer_pool_manager.cpp
//
// Identification: src/buffer/buffer_pool_manager.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "buffer/buffer_pool_manager.h"
#include <algorithm>
#include <utility>
#include <vector>
#include "buffer/arc_replacer.h"
#include "common/config.h"
#include "common/macros.h"

namespace bustub {

/**
 * @brief The constructor for a `FrameHeader` that initializes all fields to default values.
 *
 * @param frame_id The frame ID / index of the frame we are creating a header for.
 */
FrameHeader::FrameHeader(frame_id_t frame_id) : frame_id_(frame_id), data_(BUSTUB_PAGE_SIZE, 0) { Reset(); }

/** @brief Get a raw const pointer to the frame's data. */
auto FrameHeader::GetData() const -> const char * { return data_.data(); }

/** @brief Get a raw mutable pointer to the frame's data. */
auto FrameHeader::GetDataMut() -> char * { return data_.data(); }

/** @brief Resets a `FrameHeader`'s member fields. */
void FrameHeader::Reset() {
  std::fill(data_.begin(), data_.end(), 0);
  pin_count_.store(0);
  is_dirty_ = false;
  page_id_ = INVALID_PAGE_ID;
}

/**
 * @brief Creates a new `BufferPoolManager` instance and initializes all fields.
 *
 * `bpm_latch_` protects: `page_table_`, `free_frames_`, every frame's `page_id_` / `is_dirty_` bookkeeping, pin count
 * transitions together with the replacer's evictable state, and the replacer itself.
 *
 * @param num_frames The size of the buffer pool.
 * @param disk_manager The disk manager.
 * @param log_manager The log manager. Please ignore this for P1.
 */
BufferPoolManager::BufferPoolManager(size_t num_frames, DiskManager *disk_manager, LogManager *log_manager)
    : num_frames_(num_frames),
      next_page_id_(0),
      bpm_latch_(std::make_shared<std::mutex>()),
      replacer_(std::make_shared<ArcReplacer>(num_frames)),
      disk_scheduler_(std::make_shared<DiskScheduler>(disk_manager)),
      log_manager_(log_manager) {
  // Not strictly necessary...
  std::scoped_lock latch(*bpm_latch_);

  // Initialize the monotonically increasing counter at 0.
  next_page_id_.store(0);

  // Allocate all of the in-memory frames up front.
  frames_.reserve(num_frames_);

  // The page table should have exactly `num_frames_` slots, corresponding to exactly `num_frames_` frames.
  page_table_.reserve(num_frames_);

  // Initialize all of the frame headers, and fill the free frame list with all possible frame IDs (since all frames are
  // initially free).
  for (size_t i = 0; i < num_frames_; i++) {
    frames_.push_back(std::make_shared<FrameHeader>(i));
    free_frames_.push_back(static_cast<int>(i));
  }
}

/**
 * @brief Destroys the `BufferPoolManager`, freeing up all memory that the buffer pool was using.
 */
BufferPoolManager::~BufferPoolManager() = default;

/**
 * @brief Returns the number of frames that this buffer pool manager manages.
 */
auto BufferPoolManager::Size() const -> size_t { return num_frames_; }

/**
 * @brief Synchronously performs one disk read or write through the disk scheduler.
 */
void BufferPoolManager::DiskIO(bool is_write, page_id_t page_id, char *data) {
  auto promise = disk_scheduler_->CreatePromise();
  auto future = promise.get_future();
  std::vector<DiskRequest> requests;
  requests.push_back(DiskRequest{is_write, data, page_id, std::move(promise)});
  disk_scheduler_->Schedule(requests);
  future.get();
}

/**
 * @brief Writes the frame's page out to disk if it is dirty, then clears the dirty flag. Caller holds the BPM latch.
 */
void BufferPoolManager::FlushFrame(const std::shared_ptr<FrameHeader> &frame) {
  if (frame->is_dirty_) {
    DiskIO(true, frame->page_id_, frame->GetDataMut());
    frame->is_dirty_ = false;
  }
}

/**
 * @brief Finds a frame that holds no page: either from the free list, or by evicting a victim chosen by the replacer.
 * A dirty victim is written back to disk first. Caller holds the BPM latch.
 *
 * @return The id of the (reset) frame, or `std::nullopt` if every frame is pinned.
 */
auto BufferPoolManager::AcquireFrame() -> std::optional<frame_id_t> {
  if (!free_frames_.empty()) {
    frame_id_t fid = free_frames_.front();
    free_frames_.pop_front();
    return fid;
  }

  auto victim = replacer_->Evict();
  if (!victim.has_value()) {
    return std::nullopt;
  }

  frame_id_t fid = *victim;
  auto &frame = frames_[fid];
  FlushFrame(frame);
  page_table_.erase(frame->page_id_);
  frame->Reset();
  return fid;
}

/**
 * @brief Makes sure the page is in memory, pins it, and informs the replacer. Caller holds the BPM latch.
 *
 * @return The frame holding the page, or `nullptr` if no frame could be obtained.
 */
auto BufferPoolManager::PinPage(page_id_t page_id, AccessType access_type) -> std::shared_ptr<FrameHeader> {
  frame_id_t fid;
  auto it = page_table_.find(page_id);
  if (it != page_table_.end()) {
    fid = it->second;
  } else {
    auto acquired = AcquireFrame();
    if (!acquired.has_value()) {
      return nullptr;
    }
    fid = *acquired;
    auto &frame = frames_[fid];
    DiskIO(false, page_id, frame->GetDataMut());
    frame->page_id_ = page_id;
    page_table_[page_id] = fid;
  }

  auto frame = frames_[fid];
  frame->pin_count_.fetch_add(1);
  replacer_->RecordAccess(fid, page_id, access_type);
  replacer_->SetEvictable(fid, false);
  return frame;
}

/**
 * @brief Allocates a new page on disk.
 *
 * @return The page ID of the newly allocated page.
 */
auto BufferPoolManager::NewPage() -> page_id_t { return next_page_id_.fetch_add(1); }

/**
 * @brief Removes a page from the database, both on disk and in memory.
 *
 * @param page_id The page ID of the page we want to delete.
 * @return `false` if the page exists but could not be deleted, `true` if the page didn't exist or deletion succeeded.
 */
auto BufferPoolManager::DeletePage(page_id_t page_id) -> bool {
  std::scoped_lock latch(*bpm_latch_);

  auto it = page_table_.find(page_id);
  if (it != page_table_.end()) {
    frame_id_t fid = it->second;
    auto &frame = frames_[fid];
    if (frame->pin_count_.load() > 0) {
      return false;
    }
    replacer_->Remove(fid);
    page_table_.erase(it);
    frame->Reset();
    free_frames_.push_back(fid);
  }

  disk_scheduler_->DeallocatePage(page_id);
  return true;
}

/**
 * @brief Acquires an optional write-locked guard over a page of data.
 *
 * @param page_id The ID of the page we want to write to.
 * @param access_type The type of page access.
 * @return A `WritePageGuard`, or `std::nullopt` if the page cannot be brought into memory.
 */
auto BufferPoolManager::CheckedWritePage(page_id_t page_id, AccessType access_type) -> std::optional<WritePageGuard> {
  std::shared_ptr<FrameHeader> frame;
  {
    std::scoped_lock latch(*bpm_latch_);
    frame = PinPage(page_id, access_type);
    if (frame == nullptr) {
      return std::nullopt;
    }
    frame->is_dirty_ = true;
  }
  // The page latch is acquired inside the guard's constructor, after the BPM latch has been released, so that we never
  // block on a page latch while holding the BPM latch.
  return WritePageGuard(page_id, std::move(frame), replacer_, bpm_latch_, disk_scheduler_);
}

/**
 * @brief Acquires an optional read-locked guard over a page of data.
 *
 * @param page_id The ID of the page we want to read.
 * @param access_type The type of page access.
 * @return A `ReadPageGuard`, or `std::nullopt` if the page cannot be brought into memory.
 */
auto BufferPoolManager::CheckedReadPage(page_id_t page_id, AccessType access_type) -> std::optional<ReadPageGuard> {
  std::shared_ptr<FrameHeader> frame;
  {
    std::scoped_lock latch(*bpm_latch_);
    frame = PinPage(page_id, access_type);
    if (frame == nullptr) {
      return std::nullopt;
    }
  }
  return ReadPageGuard(page_id, std::move(frame), replacer_, bpm_latch_, disk_scheduler_);
}

/**
 * @brief A wrapper around `CheckedWritePage` that unwraps the inner value if it exists.
 *
 * If `CheckedWritePage` returns a `std::nullopt`, **this function aborts the entire process.**
 */
auto BufferPoolManager::WritePage(page_id_t page_id, AccessType access_type) -> WritePageGuard {
  auto guard_opt = CheckedWritePage(page_id, access_type);

  if (!guard_opt.has_value()) {
    fmt::println(stderr, "\n`CheckedWritePage` failed to bring in page {}\n", page_id);
    std::abort();
  }

  return std::move(guard_opt).value();
}

/**
 * @brief A wrapper around `CheckedReadPage` that unwraps the inner value if it exists.
 *
 * If `CheckedReadPage` returns a `std::nullopt`, **this function aborts the entire process.**
 */
auto BufferPoolManager::ReadPage(page_id_t page_id, AccessType access_type) -> ReadPageGuard {
  auto guard_opt = CheckedReadPage(page_id, access_type);

  if (!guard_opt.has_value()) {
    fmt::println(stderr, "\n`CheckedReadPage` failed to bring in page {}\n", page_id);
    std::abort();
  }

  return std::move(guard_opt).value();
}

/**
 * @brief Flushes a page's data out to disk unsafely (without taking the page latch).
 *
 * @param page_id The page ID of the page to be flushed.
 * @return `false` if the page could not be found in the page table; otherwise, `true`.
 */
auto BufferPoolManager::FlushPageUnsafe(page_id_t page_id) -> bool {
  std::scoped_lock latch(*bpm_latch_);
  auto it = page_table_.find(page_id);
  if (it == page_table_.end()) {
    return false;
  }
  FlushFrame(frames_[it->second]);
  return true;
}

/**
 * @brief Flushes a page's data out to disk safely (holding the page's read latch while writing).
 *
 * @param page_id The page ID of the page to be flushed.
 * @return `false` if the page could not be found in the page table; otherwise, `true`.
 */
auto BufferPoolManager::FlushPage(page_id_t page_id) -> bool {
  std::shared_ptr<FrameHeader> frame;
  {
    std::scoped_lock latch(*bpm_latch_);
    auto it = page_table_.find(page_id);
    if (it == page_table_.end()) {
      return false;
    }
    // Pin the frame (without counting this as a replacer access) so it cannot be evicted while we wait for the latch.
    frame = frames_[it->second];
    frame->pin_count_.fetch_add(1);
    replacer_->SetEvictable(it->second, false);
  }

  // The guard takes the read latch, flushes under the BPM latch, and unpins on destruction.
  ReadPageGuard guard(page_id, std::move(frame), replacer_, bpm_latch_, disk_scheduler_);
  guard.Flush();
  return true;
}

/**
 * @brief Flushes all page data that is in memory to disk unsafely.
 */
void BufferPoolManager::FlushAllPagesUnsafe() {
  std::scoped_lock latch(*bpm_latch_);
  for (const auto &entry : page_table_) {
    FlushFrame(frames_[entry.second]);
  }
}

/**
 * @brief Flushes all page data that is in memory to disk safely.
 */
void BufferPoolManager::FlushAllPages() {
  std::vector<page_id_t> page_ids;
  {
    std::scoped_lock latch(*bpm_latch_);
    page_ids.reserve(page_table_.size());
    for (const auto &entry : page_table_) {
      page_ids.push_back(entry.first);
    }
  }
  for (page_id_t pid : page_ids) {
    FlushPage(pid);
  }
}

/**
 * @brief Retrieves the pin count of a page. If the page does not exist in memory, return `std::nullopt`.
 *
 * @param page_id The page ID of the page we want to get the pin count of.
 * @return std::optional<size_t> The pin count if the page exists; otherwise, `std::nullopt`.
 */
auto BufferPoolManager::GetPinCount(page_id_t page_id) -> std::optional<size_t> {
  std::scoped_lock latch(*bpm_latch_);
  auto it = page_table_.find(page_id);
  if (it == page_table_.end()) {
    return std::nullopt;
  }
  return frames_[it->second]->pin_count_.load();
}

}  // namespace bustub
