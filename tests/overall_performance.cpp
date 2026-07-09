#include "v2/merge_insert.h"
#include "v2/repair_scheduler.h"
#include <unordered_set>
#include <limits>
#include <fstream>
#include <cstdlib>
#include <cstdint>

#include "log.h"
#include <index.h>
#include <cstddef>
#include <future>
#include <random>
#include <Neighbor_Tag.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

uint32_t Merge_Size = 0;
#define NUM_INSERT_THREADS 24
#define NUM_DELETE_THREADS 1
#define NUM_SEARCH_THREADS 32
#define InsertQPS 1200
#define DeleteQPS 1200

int            begin_time = 0;
diskann::Timer globalTimer;

// ---- Phase-2 scaffolding (Ask 1): durable telemetry + inert ablation flags
// ----

std::string g_cache_mode = "baseline";  // --cache-mode {baseline|filter_tenant}
std::string g_sched_mode = "fifo";      // --sched-mode {fifo|priority}
std::string g_telemetry_csv = "";       // env TELEMETRY_CSV=/path/run.csv

// query-side partition key source (mirrors insert side)

uint32_t                       g_c2_L_pred = 1;
uint32_t                       g_c2_T_cnt = 1;
std::vector<diskann::LabelId>  g_query_labels;
std::vector<diskann::TenantId> g_query_tenants;

static diskann::PartitionKey c2_query_key(size_t q) {
  diskann::PartitionKey key{};

  if (!g_query_labels.empty()) {
    key.label = (q < g_query_labels.size()) ? g_query_labels[q] : 0u;
    key.tenant = (q < g_query_tenants.size()) ? g_query_tenants[q] : 0u;
  } else {
    key.label = (diskann::LabelId)(q % g_c2_L_pred);
    key.tenant = (diskann::TenantId)(q % g_c2_T_cnt);
  }

  return key;
}

int      g_ckpt_idx = 0;
int      g_batch_idx = -1;
uint64_t g_cumulative_inserts = 0;
int      g_is_merge_boundary = 0;
double   g_c3_a1 = 1.0, g_c3_a2 = 0.0, g_c3_a3 = 1.0,
       g_c3_a4 = 0.0;  // anchor defaults
uint64_t                 g_c3_B = 0, g_c3_high_water = 0, g_c3_low_water = 0;
bool                     g_c3_sched_dr = false;
uint32_t                 g_c3_sla_default = 1;
diskann::RepairScheduler g_sched;

// C3 3a drain telemetry (published at the merge trigger, read by
// emit_telemetry_row)
size_t   g_sched_pending = 0, g_sched_drained = 0;
double   g_sched_kappa_min = 0.0, g_sched_kappa_max = 0.0;
uint64_t g_sched_max_stale_drained = 0, g_sched_max_stale_deferred = 0;

template<typename T, typename TagT>
void emit_telemetry_row(diskann::MergeInsert<T, TagT>& sync_index, float recall,
                        float p99_ms, float mean_ios, bool cal_recall) {
  if (g_telemetry_csv.empty())
    return;

  bool          need_header = !file_exists(g_telemetry_csv);
  std::ofstream ofs(g_telemetry_csv,
                    std::ios::app);  // append: survives exit(0)

  if (!ofs.is_open())
    return;

  if (need_header) {
    ofs << "timestamp,ckpt_idx,batch_idx,cumulative_inserts,merge_size,"
           "mem_points,deletion_set_total,merges_so_far,last_merge_ms,"
           "recall_at_10,p99_latency_ms,disk_deleted_ids,mean_disk_ios,"
           "cache_mode,sched_mode,io_engine,is_merge_boundary,"
           "l_pred,t_cnt,n_partitions_active,n_partitions_merging,"
           "mem_index_bytes,probed_partition_occupancy,"
           "mem_index_bytes,probed_partition_occupancy,"
           "sched_pending_tasks,sched_drained_tasks,sched_kappa_min,"
           "sched_kappa_max,sched_max_stale_drained,sched_max_stale_deferred\n";
    ;
  }

  size_t deletion_total = 0;
  for (auto& kv : sync_index._deletion_set_0)
    deletion_total += kv.second.size();
  for (auto& kv : sync_index._deletion_set_1)
    deletion_total += kv.second.size();

  size_t merges_so_far = (size_t) g_ckpt_idx + sync_index._num_merges;

  // partition telemetry (all from public members; no library change)
  auto& active_slot = (sync_index._active_index == 0) ? sync_index._mem_index_0
                                                      : sync_index._mem_index_1;

  auto& merging_slot = (sync_index._active_index == 0)
                           ? sync_index._mem_index_1
                           : sync_index._mem_index_0;

  size_t n_partitions_active = active_slot.size();
  size_t n_partitions_merging = merging_slot.size();
  size_t total_partitions = n_partitions_active + n_partitions_merging;

  // RSS proxy: each partition reallocates an Index sized 2*_merge_d (contract).
  // This is the vector-data capacity term (the tunable knob); excludes
  // graph/tag overhead, so it under-counts OS RSS but scales as #partitions x
  // 2xMerge_Size.
  size_t per_index_bytes =
      (size_t) 2 * sync_index._merge_th * sync_index._dim * sizeof(T);

  size_t mem_index_bytes = total_partitions * per_index_bytes;

  // busiest probed partition = max per-key occupancy in the active slot
  size_t probed_partition_occupancy = 0;
  for (auto& kv : sync_index._mem_points_by_key)
    if (kv.second > probed_partition_occupancy)
      probed_partition_occupancy = kv.second;

  ofs << (long long) time(nullptr) << "," << g_ckpt_idx << "," << g_batch_idx
      << "," << g_cumulative_inserts << "," << sync_index._merge_th << ","
      << sync_index._mem_points << "," << deletion_total << "," << merges_so_far
      << "," << sync_index._last_merge_ms << ","
      << (cal_recall ? recall : -1.0f) << "," << p99_ms << "," << 0
      << ","  // disk_deleted_ids: 0 under INSERTS_ONLY (lives in
              // StreamingMerger; deferred)
      << mean_ios << "," << g_cache_mode << "," << g_sched_mode << ","
      << "io_uring"
      << "," << g_is_merge_boundary << "," << sync_index._L_pred << ","
      << sync_index._T_cnt << "," << n_partitions_active << ","
      << n_partitions_merging << "," << mem_index_bytes << ","
      << probed_partition_occupancy << "," << g_sched_pending << ","
      << g_sched_drained << "," << g_sched_kappa_min << "," << g_sched_kappa_max
      << "," << g_sched_max_stale_drained << "," << g_sched_max_stale_deferred
      << std::endl;  // std::endl flushes -> row durable before any exit()

  long          vmhwm_kb = 0;
  std::ifstream stfile("/proc/self/status");
  std::string   sline;

  while (std::getline(stfile, sline))
    if (sline.rfind("VmHWM:", 0) == 0) {
      sscanf(sline.c_str(), "VmHWM: %ld kB", &vmhwm_kb);
      break;
    }

  std::cout << "peak RSS (VmHWM): " << vmhwm_kb << " KB" << std::endl;
}

// acutually also shows disk size
void ShowMemoryStatus() {
  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;

  int           tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;
  buffer.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                      1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb;

  std::cout << "memory current time: " << current_time << " RSS : " << rss
            << " KB" << std::endl;
}

std::string convertFloatToString(const float value, const int precision = 0) {
  std::stringstream stream{};
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string GetTruthFileName(std::string& truthFilePrefix, int r_start) {
  std::string fileName(truthFilePrefix);
  fileName = fileName + "/gt_" + std::to_string(r_start) + ".bin";
  LOG(INFO) << "Truth file name: " << fileName;
  return fileName;
}

template<typename T>
inline uint64_t save_bin_test(const std::string& filename, T* id, float* dist,
                              size_t npts, size_t ndims, size_t offset = 0) {
  std::ofstream writer;
  open_file_to_write(writer, filename);

  diskann::cout << "Writing bin: " << filename.c_str() << std::endl;
  writer.seekp(offset, writer.beg);
  int    npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
  writer.write((char*) &npts_i32, sizeof(int));
  writer.write((char*) &ndims_i32, sizeof(int));
  diskann::cout << "bin: #pts = " << npts << ", #dims = " << ndims
                << ", size = " << bytes_written << "B" << std::endl;

  for (int i = 0; i < npts; i++) {
    for (int j = 0; j < ndims; j++) {
      writer.write((char*) (id + i * ndims + j), sizeof(T));
      writer.write((char*) (dist + i * ndims + j), sizeof(float));
    }
  }
  writer.close();
  diskann::cout << "Finished writing bin." << std::endl;
  return bytes_written;
}

template<typename T, typename TagT>
void sync_search_kernel(T* query, size_t query_num, size_t query_aligned_dim,
                        const int recall_at, _u64 L,
                        diskann::MergeInsert<T, TagT>& sync_index,
                        std::string& truthset_file, bool merged,
                        bool calRecall) {
  unsigned* gt_ids = NULL;
  float*    gt_dists = NULL;
  size_t    gt_num, gt_dim;

  if (!file_exists(truthset_file)) {
    calRecall = false;
  }

  if (calRecall) {
    std::cout << "current truthfile: " << truthset_file << std::endl;
    diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
  }

  float* query_result_dists = new float[recall_at * query_num];
  TagT*  query_result_tags = new TagT[recall_at * query_num];

  for (_u32 q = 0; q < query_num; q++) {
    for (_u32 r = 0; r < (_u32) recall_at; r++) {
      query_result_tags[q * recall_at + r] = std::numeric_limits<TagT>::max();
      query_result_dists[q * recall_at + r] = std::numeric_limits<float>::max();
    }
  }

  std::vector<double> latency_stats(query_num, 0);
  diskann::QueryStats stats[query_num];
  std::string         recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18)
            << "Mean Lat" << std::setw(12) << "50 Lat" << std::setw(12)
            << "90 Lat" << std::setw(12) << "95 Lat" << std::setw(12)
            << "99 Lat" << std::setw(12) << "99.9 Lat" << std::setw(12)
            << recall_string << std::setw(12) << "Disk IOs" << std::endl;
  std::cout << "==============================================================="
               "==============="
            << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS) schedule(dynamic)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
    auto qs = std::chrono::high_resolution_clock::now();
    // stats[i].n_current_used = 8;
    diskann::PartitionKey qk = (g_cache_mode == "filter-tenant")
                                   ? c2_query_key((size_t) i)
                                   : diskann::PartitionKey{};

    sync_index.search_sync(query + i * query_aligned_dim, recall_at, L,
                           query_result_tags + i * recall_at,
                           query_result_dists + i * recall_at, stats + i, qk);

    auto qe = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = qe - qs;
    latency_stats[i] = diff.count() * 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
  float                         qps = (query_num / diff.count());
  float                         recall = 0;

  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;
  if (calRecall) {
    // for (int i = 0; i < recall_at; ++i) {
    //   LOG(INFO) << query_result_tags[i] << " " << gt_ids[i];
    // }
    recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                       query_result_tags, recall_at, recall_at);
    {
      const char* p3dir = std::getenv("P3_DUMP");
      if (p3dir != nullptr) {
        std::string fn =
            std::string(p3dir) + "/L_" + std::to_string((long long) L) + ".u32";

        std::ofstream ofs(fn, std::ios::binary);

        uint32_t nq = (uint32_t) query_num;
        uint32_t kk = (uint32_t) recall_at;

        ofs.write((char*) &nq, 4);
        ofs.write((char*) &kk, 4);

        ofs.write((char*) query_result_tags,
                  (size_t) query_num * (size_t) recall_at * sizeof(TagT));

        ofs.close();

        std::cout << "P3 dump written: " << fn << std::endl;
      }
    }

    delete[] gt_ids;
    gt_ids = nullptr;
  }

  std::cout << "search current time: " << current_time << std::endl;

  float mean_ios = (float) diskann::get_mean_stats(
      stats, query_num,
      [](const diskann::QueryStats& stats) { return stats.n_ios; });

  std::sort(latency_stats.begin(), latency_stats.end());
  std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
            << ((float) std::accumulate(latency_stats.begin(),
                                        latency_stats.end(), 0)) /
                   (float) query_num
            << std::setw(12)
            << (float) latency_stats[(_u64) (0.50 * ((double) query_num))]
            << std::setw(12)
            << (float) latency_stats[(_u64) (0.90 * ((double) query_num))]
            << std::setw(12)
            << (float) latency_stats[(_u64) (0.95 * ((double) query_num))]
            << std::setw(12)
            << (float) latency_stats[(_u64) (0.99 * ((double) query_num))]
            << std::setw(12)
            << (float) latency_stats[(_u64) (0.999 * ((double) query_num))]
            << std::setw(12) << recall << std::setw(12) << mean_ios
            << std::endl;
  float p99_ms =
      (query_num == 0)
          ? 0.0f
          : (float) latency_stats[(_u64) (0.99 * ((double) query_num))];
  emit_telemetry_row(sync_index, recall, p99_ms, mean_ios, calRecall);
  delete[] query_result_dists;
  delete[] query_result_tags;
}

template<typename T, typename TagT>
void merge_kernel(diskann::MergeInsert<T, TagT>& sync_index,
                  std::string&                   save_path) {
  // sync_index.dummy_merge();
  //   ShowMemoryStatus();
  sync_index.final_merge();
}

template<typename T, typename TagT>
void deletion_kernel(T* data_load, diskann::MergeInsert<T, TagT>& sync_index,
                     std::vector<TagT>& delete_vec, size_t aligned_dim) {
  if (std::getenv("INSERTS_ONLY") != nullptr) {
    std::cout << "INSERTS_ONLY mode: skipping deletion kernel" << std::endl;
    return;
  }

  diskann::Timer      timer;
  size_t              npts = delete_vec.size();
  std::vector<double> delete_latencies(npts, 0);
  std::cout << "Begin Delete" << std::endl;
  std::atomic_size_t success(0);
#pragma omp parallel for num_threads(NUM_DELETE_THREADS)
  for (_s64 i = 0; i < (_s64) delete_vec.size(); i++) {
    diskann::Timer      delete_timer;
    diskann::QueryStats stats;
    sync_index.lazy_delete(delete_vec[i]);
    success++;
    delete_latencies[i] = ((double) delete_timer.elapsed());

    if (success == DeleteQPS) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      success = 0;
    }
  }
  float time_secs = timer.elapsed() / 1.0e6f;
  std::sort(delete_latencies.begin(), delete_latencies.end());
  // std::cout << "Deleted " << delete_vec.size() << " / " << success.load() <<
  // " points in " << time_secs << "s"
  //           << std::endl;
  std::cout << "Mem index deletion time : " << timer.elapsed() / 1000 << " ms"
            << std::endl
            << "10th percentile deletion time : "
            << delete_latencies[(size_t) (0.10 * ((double) npts))]
            << " microsec" << std::endl
            << "50th percentile deletion time : "
            << delete_latencies[(size_t) (0.5 * ((double) npts))] << " microsec"
            << std::endl
            << "90th percentile deletion time : "
            << delete_latencies[(size_t) (0.90 * ((double) npts))]
            << " microsec" << std::endl
            << "99th percentile deletion time : "
            << delete_latencies[(size_t) (0.99 * ((double) npts))]
            << " microsec" << std::endl
            << "99.9th percentile deletion time : "
            << delete_latencies[(size_t) (0.999 * ((double) npts))]
            << " microsec" << std::endl;
}

template<typename T, typename TagT>
void insertion_kernel(T* data_load, diskann::MergeInsert<T, TagT>& sync_index,
                      std::vector<TagT>& insert_vec, size_t aligned_dim) {
  diskann::Timer      timer;
  size_t              npts = insert_vec.size();
  std::vector<double> insert_latencies(npts, 0);
  std::cout << "Begin Insert" << std::endl;
  std::atomic_size_t success(0);

  auto st = std::chrono::high_resolution_clock::now();

  auto getQPS = [&]() {
    auto cur = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(cur - st);
    if (dur.count() == 0) {
      return 0ul;
    }
    return (1000 * success) / dur.count();
  };

#pragma omp parallel for num_threads(2)
  for (_s64 i = 0; i < (_s64) insert_vec.size(); i++) {
    diskann::Timer insert_timer;
    sync_index.insert(data_load + aligned_dim * i, insert_vec[i]);
    ++success;
    insert_latencies[i] = ((double) insert_timer.elapsed());
    while (getQPS() >= InsertQPS) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  float time_secs = timer.elapsed() / 1.0e6f;
  std::sort(insert_latencies.begin(), insert_latencies.end());
  std::cout << "Inserted " << insert_vec.size() << " points in " << time_secs
            << "s" << std::endl;
  std::cout << "Mem index insertion time : " << timer.elapsed() / 1000 << " ms"
            << std::endl
            << "10th percentile insertion time : "
            << insert_latencies[(size_t) (0.10 * ((double) npts))]
            << " microsec" << std::endl
            << "50th percentile insertion time : "
            << insert_latencies[(size_t) (0.5 * ((double) npts))] << " microsec"
            << std::endl
            << "90th percentile insertion time : "
            << insert_latencies[(size_t) (0.90 * ((double) npts))]
            << " microsec" << std::endl
            << "99th percentile insertion time : "
            << insert_latencies[(size_t) (0.99 * ((double) npts))]
            << " microsec" << std::endl
            << "99.9th percentile insertion time : "
            << insert_latencies[(size_t) (0.999 * ((double) npts))]
            << " microsec" << std::endl;
}

template<typename T, typename TagT = uint32_t>
void get_trace(std::string data_bin, uint64_t l_start, uint64_t r_start,
               uint64_t n, std::vector<TagT>& delete_tags,
               std::vector<TagT>& insert_tags, std::vector<T>& data_load) {
  for (uint64_t i = l_start; i < l_start + n; ++i)
    delete_tags.push_back(i);

  for (uint64_t i = r_start; i < r_start + n; ++i)
    insert_tags.push_back(i);

  int           npts_i32, dim_i32;
  std::ifstream reader(data_bin, std::ios::binary | std::ios::ate);

  reader.seekg(0, reader.beg);
  reader.read((char*) &npts_i32, sizeof(int));
  reader.read((char*) &dim_i32, sizeof(int));

  size_t data_dim = dim_i32;
  data_load.resize(n * data_dim);

  reader.seekg(2 * sizeof(int) + r_start * data_dim * sizeof(T), reader.beg);
  reader.read((char*) data_load.data(), sizeof(T) * n * data_dim);

  // ---- Task 1 (2B.5): seeded insert-order permutation, default OFF ----
  // C2_SEED unset  => no permutation => byte-identical to today (anchor safe).
  // C2_SEED=<n>    => deterministic per-(seed,window) shuffle of the
  //                  (tag,vector) pairs. Same point set at every res
  //                  (gt_<res>.bin still matches), different realized order.
  //

  if (const char* se = std::getenv("C2_SEED")) {
    uint64_t        seed = std::strtoull(se, nullptr, 10);
    std::mt19937_64 rng(seed ^ (0x9E3779B97F4A7C15ULL * (r_start + 1)));

    std::vector<uint64_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0ull);

    for (uint64_t i = n; i-- > 1;) {
      std::uniform_int_distribution<uint64_t> d(0, i);
      std::swap(perm[i], perm[d(rng)]);
    }

    std::vector<TagT> it2(n);
    std::vector<T>    dl2(n * data_dim);

    for (uint64_t i = 0; i < n; ++i) {
      it2[i] = insert_tags[perm[i]];

      std::memcpy(dl2.data() + i * data_dim,
                  data_load.data() + perm[i] * data_dim, data_dim * sizeof(T));
    }

    insert_tags.swap(it2);
    data_load.swap(dl2);

    uint64_t h = 1469598103934665603ULL;  // FNV-1a over realized tag order

    for (uint64_t i = 0; i < n; ++i) {
      uint64_t v = (uint64_t) insert_tags[i];
      for (int b = 0; b < 8; ++b) {
        h ^= (v & 0xff);
        h *= 1099511628211ULL;
        v >>= 8;
      }
    }

    std::cout << "[C2_SEED] seed=" << seed << " window_r_start=" << r_start
              << " n=" << n << " realized_order_fnv1a=0x" << std::hex << h
              << std::dec << std::endl;
  }
}

template<typename T, typename TagT>
void update(const std::string& data_bin, const unsigned L_disk,
            const unsigned R_disk, const float alpha_disk, int step,
            const size_t base_num, const unsigned nodes_to_cache,
            std::string& save_path, const std::string& query_file,
            std::string& truthset_file, const int recall_at,
            std::vector<_u64> Lsearch, const unsigned beam_width,
            diskann::Distance<T>* dist_cmp, int ckpt = 0) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L_mem", 128);
  paras.Set<unsigned>("R_mem", 64);
  paras.Set<float>("alpha_mem", 1.2);
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", R_disk);
  paras.Set<float>("alpha_disk", alpha_disk);
  paras.Set<unsigned>("C", 384);
  paras.Set<unsigned>("beamwidth", beam_width);
  paras.Set<unsigned>("nodes_to_cache", 0);
  paras.Set<unsigned>("num_search_threads",
                      256);  // more for less contention of thread data.
  std::vector<T> data_load;
  size_t         dim{}, aligned_dim{};

  diskann::Timer timer;

  std::cout << "Loading queries " << std::endl;
  T*     query = NULL;
  size_t query_num, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  dim = query_dim;
  aligned_dim = query_aligned_dim;
  diskann::Metric metric = diskann::Metric::L2;

  std::cout << "Searching before inserts: " << std::endl;

  int               batch = step * 10;  // step / 10;
  int               inMemorySize = 0;
  std::future<void> merge_future;

  fprintf(stderr, "Need copy: %d, need swap: %d\n", ckpt == 0, ckpt % 2 != 0);
  diskann::MergeInsert<T, TagT> sync_index(
      paras, dim, save_path + "_mem", save_path, save_path + "_merge", dist_cmp,
      metric, false, save_path, ckpt == 0, ckpt % 2 != 0, 0);

  uint64_t npts_modulo = sync_index._disk_index->num_points;
  uint64_t x = 1;
  while (npts_modulo > 1) {
    x *= 10;
    npts_modulo /= 10;
  }

  uint64_t index_npts = x;
  uint64_t vecs_per_step = index_npts / step;
  double   merge_ratio = ((double) Merge_Size / index_npts);

  uint64_t per_ckpt_inserts = (g_sched_mode == "priority")
                                  ? (uint64_t) g_c3_high_water
                                  : (uint64_t) Merge_Size;

  uint64_t ckpt_i = 0, res = 0, tmp_npts = index_npts;
  for (int i = 0; i < ckpt; ++i) {
    res += per_ckpt_inserts;
    ckpt_i += per_ckpt_inserts / vecs_per_step;
    tmp_npts += per_ckpt_inserts;
    // Merge_Size =
    //     ((uint32_t) (merge_ratio * tmp_npts)) / vecs_per_step *
    //     vecs_per_step;
  }
  sync_index.init_mem_index(Merge_Size);
  // configure write-path partition routing
  {
    uint32_t c2_L = 1, c2_T = 1;

    if (const char* e = std::getenv("C2_L_PRED"))
      c2_L = (uint32_t) std::atoi(e);

    if (const char* e = std::getenv("C2_T_CNT"))
      c2_T = (uint32_t) std::atoi(e);

    sync_index.set_synthetic_partitioning(c2_L, c2_T);

    const char* lf = std::getenv("C2_LABELS_FILE");
    const char* tf = std::getenv("C2_TENANTS_FILE");

    if (lf || tf)
      sync_index.load_metadata(lf ? lf : "", tf ? tf : "");

    // query-side key source (mirror of the insert side)
    g_c2_L_pred = c2_L;
    g_c2_T_cnt = c2_T;

    if (const char* qlf = std::getenv("C2_QUERY_LABELS_FILE")) {
      std::ifstream f(qlf);
      std::string   line;
      while (std::getline(f, line))
        g_query_labels.push_back(
            line.empty() ? 0u : (diskann::LabelId) std::stoul(line));
    }

    if (const char* qtf = std::getenv("C2_QUERY_TENANTS_FILE")) {
      std::ifstream f(qtf);
      std::string   line;
      while (std::getline(f, line))
        g_query_tenants.push_back(
            line.empty() ? 0u : (diskann::TenantId) std::stoul(line));
    }

    if (g_cache_mode == "filter-tenant") {
      if (!g_query_labels.empty())
        std::cout << "[C2] query key source: sidecar (" << g_query_labels.size()
                  << " labels, " << g_query_tenants.size() << " tenants)"
                  << std::endl;
      else
        std::cout << "[C2] query key source: synthetic label = q % "
                  << g_c2_L_pred << ", tenant = q % " << g_c2_T_cnt
                  << std::endl;
    }
  }

  // C3 3b: restore scheduler queue + deferred staged partitions across restart.
  if (g_sched_mode == "priority") {
    if (const char* sf = std::getenv("SCHED_STATE_FILE")) {
      std::ifstream sfs(sf);
      if (sfs.is_open()) {
        std::stringstream ss;
        ss << sfs.rdbuf();

        g_sched.deserialize(ss.str());

        // env config wins over persisted config (workload state = tasks only):
        g_sched.configure(g_c3_a1, g_c3_a2, g_c3_a3, g_c3_a4, (size_t) g_c3_B,
                          (size_t) g_c3_high_water, (size_t) g_c3_low_water,
                          g_c3_sched_dr);

        std::cout << "[C3-3b] restored scheduler queue: "
                  << g_sched.pending_count() << " pending task(s)" << std::endl;
      }

      // reload deferred staged points into the active slot (no-op if absent)
      sync_index.load_deferred_staging(std::string(sf) + ".staging");
    }
  }

  LOG(INFO) << "index npts: " << index_npts
            << " vecs per step: " << vecs_per_step << " ckpt_i: " << ckpt_i
            << " merge ratio: " << merge_ratio << " res: " << res;
  LOG(INFO) << "Merge size is set to " << Merge_Size;
  // exit(0);

  if (ckpt_i == 0 && ckpt != 0) {
    std::cerr << "ckpt_i is 0, please check ckpt and Merge_Size" << std::endl;
    exit(-1);
  }

  std::string currentFileName = GetTruthFileName(truthset_file, res);
  begin_time = globalTimer.elapsed() / 1.0e6f;
  ShowMemoryStatus();
  g_ckpt_idx = ckpt;  // phase 2 telemetry
  g_batch_idx = -1;   // pre-insert baseline / post merge snapshot
  g_cumulative_inserts = res;
  g_is_merge_boundary =
      (ckpt != 0) ? 1 : 0;  // relaunched proc -> post merge recall row
  for (int i = 0; i < Lsearch.size(); ++i) {
    sync_search_kernel(query, query_num, query_aligned_dim, recall_at,
                       Lsearch[i], sync_index, currentFileName, false, true);
  }

  g_is_merge_boundary = 0;

  for (int i = ckpt_i; i < batch; i++) {
    g_batch_idx = i;  // phase 2 telemetry
    std::cout << "Batch: " << i << " Total Batch : " << batch << std::endl;
    std::vector<unsigned> insert_vec;
    std::vector<unsigned> delete_vec;

    /**Prepare for update*/
    uint64_t st = vecs_per_step * i;
    get_trace<T, TagT>(data_bin, st, st + index_npts, vecs_per_step, delete_vec,
                       insert_vec, data_load);

    std::future<void> insert_future = std::async(
        std::launch::async, insertion_kernel<T, TagT>, data_load.data(),
        std::ref(sync_index), std::ref(insert_vec), aligned_dim);

    std::future<void> delete_future = std::async(
        std::launch::async, deletion_kernel<T, TagT>, data_load.data(),
        std::ref(sync_index), std::ref(delete_vec), aligned_dim);

    int                total_queries = 0;
    std::future_status insert_status;
    std::future_status delete_status;
    do {
      insert_status = insert_future.wait_for(std::chrono::seconds(5));
      delete_status = delete_future.wait_for(std::chrono::seconds(5));
      if (insert_status == std::future_status::deferred ||
          delete_status == std::future_status::deferred) {
        std::cout << "deferred\n";
      } else if (insert_status == std::future_status::timeout ||
                 delete_status == std::future_status::timeout) {
        ShowMemoryStatus();
        sync_search_kernel(query, query_num, query_aligned_dim, recall_at,
                           Lsearch[0], sync_index, currentFileName, false,
                           false);
        total_queries += query_num;
        std::cout << "Queries processed: " << total_queries << std::endl;
      }
      if (insert_status == std::future_status::ready) {
        std::cout << "Insertions complete!\n";
      }
      if (delete_status == std::future_status::ready) {
        std::cout << "Deletions complete!\n";
      }
    } while (insert_status != std::future_status::ready ||
             delete_status != std::future_status::ready);

    inMemorySize += insert_vec.size();

    std::cout << "Search after update, current vector number: " << res
              << std::endl;

    res += vecs_per_step;
    g_cumulative_inserts = res;  // phase 2 telemetry

    // C3 (Agent 2): populate the scheduler from the C2 per-partition occupancy
    // map. s_w = current cumulative-insert count (logical time); push_or_update
    // keeps the EARLIEST s_w, so a partition retains its create time. One
    // upsert per touched key per batch. Only meaningful under priority.
    if (g_sched_mode == "priority") {
      for (auto& kv : sync_index._mem_points_by_key) {
        if (kv.second == 0)
          continue;  // skip drained/empty partitions

        diskann::RepairTask t;
        t.key = kv.first;
        t.occupancy = kv.second;
        t.dr_proxy = (float) kv.second;     // occupancy proxy (dr term gated)
        t.s_w = res;                        // current count; earliest kept
        t.tightest_sla = g_c3_sla_default;  // uniform this ask

        g_sched.push_or_update(t);
      }
    }

    currentFileName = GetTruthFileName(truthset_file, res);
    ShowMemoryStatus();

    for (int i = 0; i < Lsearch.size(); ++i) {
      sync_search_kernel(query, query_num, query_aligned_dim, recall_at,
                         Lsearch[i], sync_index, currentFileName, false, true);
    }

    if (i == batch - 1) {
      std::cout << "Done" << std::endl;
      exit(0);
    } else if (inMemorySize >= (int) ((g_sched_mode == "priority")
                                          ? g_c3_high_water
                                          : (uint64_t) Merge_Size)) {
      std::cout << " Begin Merge" << std::endl;

      // C3 (Agent 2): priority delegation. Under default config
      // (B = Merge_Size, high_water = Merge_Size, uniform SLA) select_batch
      // returns ALL pending partitions => same combined merge as FIFO. The
      // ACTUAL drain stays the unchanged C2 combined single-merge (final_merge
      // drains all partitions of the active slot); the selective subset
      // that would honor a partial "selected" set is Agent 3 (needs
      // merge_insert.*). We do NOT modify merge_insert here (J1 preserved).
      if (g_sched_mode == "priority") {
        uint64_t t_now = res;  // cumulative-insert count = logical time

        // Stats over the FULL pending set at t_now (before draining).
        std::vector<diskann::RepairTask> snap = g_sched.snapshot();

        double kmin = 0.0;
        double kmax = 0.0;

        for (size_t si = 0; si < snap.size(); ++si) {
          double k = g_sched.kappa(snap[si], t_now);
          if (si == 0) {
            kmin = kmax = k;
          } else {
            if (k < kmin)
              kmin = k;
            if (k > kmax)
              kmax = k;
          }
        }

        std::vector<diskann::RepairTask> selected =
            g_sched.select_batch(t_now, (size_t) g_c3_B);

        auto packk = [](const diskann::PartitionKey& k) {
          return ((uint64_t) k.label << 32) ^ (uint64_t) k.tenant;
        };

        std::unordered_set<uint64_t>       sel_ids;
        std::vector<diskann::PartitionKey> selkeys;

        uint64_t max_stale_drained = 0, max_stale_deferred = 0;

        for (auto& t : selected) {
          sel_ids.insert(packk(t.key));
          selkeys.push_back(t.key);

          uint64_t st = (t_now >= t.s_w) ? (t_now - t.s_w) : 0;
          if (st > max_stale_drained)
            max_stale_drained = st;
        }

        for (auto& t : snap) {
          if (sel_ids.count(packk(t.key)))
            continue;

          uint64_t st = (t_now >= t.s_w) ? (t_now - t.s_w) : 0;
          if (st > max_stale_deferred)
            max_stale_deferred = st;
        }

        g_sched_pending = snap.size();
        g_sched_drained = selected.size();
        g_sched_kappa_min = snap.empty() ? 0.0 : kmin;
        g_sched_kappa_max = snap.empty() ? 0.0 : kmax;
        g_sched_max_stale_drained = max_stale_drained;
        g_sched_max_stale_deferred = max_stale_deferred;

        sync_index.set_merge_selection(selkeys);  // gate the merge to selected

        std::cout << "[C3] drain t_now=" << t_now << " B=" << g_c3_B
                  << " pending=" << g_sched_pending
                  << " drained=" << g_sched_drained << " kappa=["
                  << g_sched_kappa_min << "," << g_sched_kappa_max << "]"
                  << " max_stale_drained=" << g_sched_max_stale_drained
                  << " max_stale_deferred=" << g_sched_max_stale_deferred
                  << std::endl;
      }

      merge_future = std::async(std::launch::async, merge_kernel<T, TagT>,
                                std::ref(sync_index), std::ref(save_path));

      std::this_thread::sleep_for(std::chrono::seconds(5));
      std::cout << " Sending Merge" << std::endl;

      inMemorySize = 0;
      std::future_status merge_status;
      do {
        merge_status = merge_future.wait_for(std::chrono::seconds(10));
        ShowMemoryStatus();
        sync_search_kernel(query, query_num, query_aligned_dim, recall_at,
                           Lsearch[0], sync_index, currentFileName, false,
                           false);
      } while (merge_status != std::future_status::ready);
      ShowMemoryStatus();
      g_is_merge_boundary = 1;  // phase 2 telemetry
      emit_telemetry_row(sync_index, -1.0f, 0.0f, 0.0f,
                         false);  // post merge snapshot
      g_is_merge_boundary = 0;

      if (g_sched_mode == "priority" &&
          std::getenv("C3_INPROC_PROBE") != nullptr) {
        std::cout << "[C3] in-process deferred-served probe (post-drain,"
                  << " pre-exit)" << std::endl;

        // Deferred partitions are still staged in the inactive slot here;
        // search_sync probes BOTH slots (unchanged C2 path) => this recall
        // reflects deferred contribution. Compare vs the same run with
        // C3_HIGH_WATER=Merge_Size (no deferral) to see they still serve.
        sync_search_kernel(query, query_num, query_aligned_dim, recall_at,
                           Lsearch[0], sync_index, currentFileName, false,
                           true);
      }

      // C3 3b: persist deferred scheduler queue + staged points before exit(0).
      // Guarded on "something deferred" => default/anchor writes nothing.
      if (g_sched_mode == "priority") {
        if (const char* sf = std::getenv("SCHED_STATE_FILE")) {
          if (g_sched.pending_count() > 0) {  // deferred tasks remain
            sync_index.save_deferred_staging(std::string(sf) + ".staging");

            std::ofstream os(sf, std::ios::trunc);
            os << g_sched
                      .serialize();  // holds only deferred after select_batch
            os.close();

            std::cout << "[C3-3b] persisted scheduler ("
                      << g_sched.pending_count()
                      << " deferred task(s)) + staging before exit"
                      << std::endl;
          }
        }
      }

      std::cout << "Merge finished for checkpoint " << ckpt;

      exit(0);  // wait for reboot.
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 15) {
    std::cout
        << "Correct usage: " << argv[0]
        << " <type[int8/uint8/float]> <data_bin> <L_disk> <R_disk> <alpha_disk>"
        << " <num_start> <#nodes_to_cache>"
        << " <indice_path> <query_file> <truthset_prefix> <recall@>"
        << " <#beam_width>"
        << " <step> <ckpt> <Lsearch> <L2>" << std::endl;
    exit(-1);
  }

  int         arg_no = 2;
  std::string data_bin = std::string(argv[arg_no++]);
  unsigned    L_disk = (unsigned) atoi(argv[arg_no++]);
  unsigned    R_disk = (unsigned) atoi(argv[arg_no++]);
  float       alpha_disk = (float) std::atof(argv[arg_no++]);
  size_t      num_start = (size_t) std::atoi(argv[arg_no++]);
  unsigned    nodes_to_cache = (unsigned) std::atoi(argv[arg_no++]);
  std::string save_path(argv[arg_no++]);

  std::string query_file(argv[arg_no++]);
  std::string truthset(argv[arg_no++]);
  int         recall_at = (int) std::atoi(argv[arg_no++]);
  unsigned    beam_width = (unsigned) std::atoi(argv[arg_no++]);
  int         step = (int) std::atoi(argv[arg_no++]);
  int         ckpt = (int) std::atoi(argv[arg_no++]);
  Merge_Size = (unsigned) std::atoi(argv[arg_no++]);
  std::cerr << "Merge size: " << Merge_Size << std::endl;

  std::vector<uint64_t> Lsearch;

  for (int i = arg_no; i < argc; ++i) {
    std::string a = argv[i];

    if (a == "--cache-mode" && i + 1 < argc) {
      g_cache_mode = argv[++i];
    } else if (a == "--sched-mode" && i + 1 < argc) {
      g_sched_mode = argv[++i];
    } else if (a == "--sched-dr") {  // C3: enable dr term (flag, no value)
      g_c3_sched_dr = true;
    } else {
      Lsearch.push_back(std::atoi(argv[i]));
    }
  }

  for (auto& x : Lsearch) {
    std::cerr << "Lsearch: " << x << std::endl;
  }

  // phase2: resolve inert ablation flags + telemetry sink (no behavior change)
  if (g_cache_mode != "baseline" && g_cache_mode != "filter-tenant") {
    std::cerr << "Unknown --cache-mode '" << g_cache_mode << "', using baseline"
              << std::endl;
    g_cache_mode = "baseline";
  }

  if (g_sched_mode != "fifo" && g_sched_mode != "priority") {
    std::cerr << "Unknown --sched-mode '" << g_sched_mode << "', using fifo"
              << std::endl;
    g_sched_mode = "fifo";
  }

  if (g_cache_mode == "filter-tenant")
    std::cout << "filter-tenant: query-path routing active" << std::endl;

  // scheduler config from env, contract defaults when unset.
  // Merge_Size is already parsed above, so B/high_water default to it.
  {
    auto envd = [](const char* k, double d) {
      const char* e = std::getenv(k);
      return e ? std::atof(e) : d;
    };

    auto envu = [](const char* k, uint64_t d) {
      const char* e = std::getenv(k);
      return e ? (uint64_t) std::strtoull(e, nullptr, 10) : d;
    };

    g_c3_a1 = envd("C3_ALPHA1", 1.0);
    g_c3_a2 = envd("C3_ALPHA2", 0.0);
    g_c3_a3 = envd("C3_ALPHA3", 1.0);
    g_c3_a4 = envd("C3_ALPHA4", 0.0);

    g_c3_B = envu("C3_B", (uint64_t) Merge_Size);
    g_c3_high_water = envu("C3_HIGH_WATER", (uint64_t) Merge_Size);
    g_c3_low_water = envu("C3_LOW_WATER", 0);

    g_c3_sla_default =
        (uint32_t) envu("C3_SLA_DEFAULT", 1);  // uniform => insert
    // C3_SLA_TIERS / C3_SLA_FILE reserved for Agent 3 / 2E - NOT parsed here.

    g_sched.configure(g_c3_a1, g_c3_a2, g_c3_a3, g_c3_a4, (size_t) g_c3_B,
                      (size_t) g_c3_high_water, (size_t) g_c3_low_water,
                      g_c3_sched_dr);
  }

  if (g_sched_mode == "priority")
    std::cout << "[C3] sched_mode=priority alpha=(" << g_c3_a1 << "," << g_c3_a2
              << "," << g_c3_a3 << "," << g_c3_a4 << ")"
              << " B=" << g_c3_B << " high_water=" << g_c3_high_water
              << " low_water=" << g_c3_low_water
              << " sched_dr=" << (g_c3_sched_dr ? "on" : "off")
              << " sla_default=" << g_c3_sla_default << std::endl;

  if (const char* tcsv = std::getenv("TELEMETRY_CSV"))
    g_telemetry_csv = tcsv;

  std::cout << "[phase2] cache_mode=" << g_cache_mode
            << " sched_mode=" << g_sched_mode << " telemetry_csv="
            << (g_telemetry_csv.empty() ? std::string("<none>")
                                        : g_telemetry_csv)
            << " io_engine=io_uring" << std::endl;

  if (std::string(argv[1]) == std::string("int8")) {
    diskann::DistanceL2Int8 dist_cmp;
    update<int8_t, unsigned>(data_bin, L_disk, R_disk, alpha_disk, step,
                             num_start, nodes_to_cache, save_path, query_file,
                             truthset, recall_at, Lsearch, beam_width,
                             &dist_cmp, ckpt);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    diskann::DistanceL2UInt8 dist_cmp;
    update<uint8_t, unsigned>(data_bin, L_disk, R_disk, alpha_disk, step,
                              num_start, nodes_to_cache, save_path, query_file,
                              truthset, recall_at, Lsearch, beam_width,
                              &dist_cmp, ckpt);
  } else if (std::string(argv[1]) == std::string("float")) {
    diskann::DistanceL2 dist_cmp;
    update<float, unsigned>(data_bin, L_disk, R_disk, alpha_disk, step,
                            num_start, nodes_to_cache, save_path, query_file,
                            truthset, recall_at, Lsearch, beam_width, &dist_cmp,
                            ckpt);
  } else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
