#pragma once

// SLA-aware composite-key repair scheduler.
//
// Header-only priority queue over staged (label,tenant) partitions. The driver
// consults this at the merge trigger to decide WHICH partitions fill
// the next combined merge and IN WHAT ORDER, scored by a composite key kappa.
// This file is pure library logic : no driver wiring, no merge path.

// -----------------------------------------------------------------------------
// kappa normalization (frozen shape; exact form documented here ):
//
// kappa(t) = a1*stale_n + a2*dr_n + a3*sla_n + a4*mig_n
//
// Raw per-term values for a task x:
//
// stale(x)   = (t_now >= x.s_w) ? (t_now - x.s_w) : 0      // logical age
// dr(x)      = x.dr_proxy
// sla_raw(x) = (double)x.tightest_sla                     // tighter SLO
//                                                      // (smaller value)
//                                                      // => larger raw
//                                                      // => higher prio
// mig(x)     = 0                                        // Phase-3 stub
//
// Each term is MIN-MAX normalized to [0,1] across the CURRENT pending set at
// call time:
//     f_n(x) = (max_f == min_f) ? 0.0 : (f(x) - min_f) / (max_f - min_f)
//
// If a term has a single distinct value across the pending set (e.g. uniform
// SLA, or mig which is always 0), its _n is defined as 0 => that term is inert.
// The dr term is additionally gated by dr_enabled (contributes only when the
// --sched-dr lever is on AND a2 > 0).
//
// Defaults (=> anchor collapses to FIFO): a1=1, a2=0, a3=1, a4=0,
// dr_enabled=false. Under uniform SLA and defaults, ordering is driven purely
// by staleness (older s_w first) -> FIFO.
// -----------------------------------------------------------------------------

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "v2/partition_key.h"

namespace diskann {

struct RepairTask {
    PartitionKey key;

    uint64_t s_w = 0;           // logical create time = cumulative-insert
                                // count at first touch by this cycle

    uint32_t tightest_sla = 0;  // tightest SLO across key.tenant's set
    size_t occupancy = 0;       // points staged in this partition
    float dr_proxy = 0.0f;      // occupancy (or occupancy^2); dr term only
    uint64_t ceiling = std::numeric_limits<uint64_t>::max(); 
};

class RepairScheduler {
	
	public:
    void configure(double a1, double a2, double a3, double a4,
                   size_t B, size_t high_water, size_t low_water,
                   bool dr_enabled) {
        _a1 = a1;
        _a2 = a2;
        _a3 = a3;
        _a4 = a4;

        _B = B;
        _high_water = high_water;
        _low_water = low_water;

        _dr_enabled = dr_enabled;
    }

    // Upsert by key. Keep the EARLIEST s_w (a partition touched again does not
    // reset its logical create time); refresh occupancy/dr_proxy/tightest_sla.
    void push_or_update(const RepairTask& t) {
        auto it = _pending.find(t.key);

        if (it == _pending.end()) {
            _pending.emplace(t.key, t);
        } else {
            RepairTask& cur = it->second;

            cur.s_w = std::min(cur.s_w, t.s_w);   // earliest wins
            cur.occupancy = t.occupancy;
            cur.dr_proxy = t.dr_proxy;
            cur.tightest_sla = t.tightest_sla;
            cur.ceiling = t.ceiling;
        }
    }

    // Score every pending task by kappa(., t_now) descending; accumulate tasks
    // until summed occupancy >= B (smallest such prefix), or return ALL pending
    // if their total occupancy <= B. Retain the non-selected tasks in the queue.
    // Deterministic tie-break: equal kappa => order by PartitionKey (label, then
    // tenant) ascending.
	
    std::vector<RepairTask> select_batch(uint64_t t_now, size_t B) {
    std::vector<RepairTask> all;
    all.reserve(_pending.size());
    for (const auto& kv : _pending)
        all.push_back(kv.second);

    // Precompute kappa once (normalization is over the fixed pending set).
    TermStats st = compute_stats(t_now);
    std::vector<std::pair<double, size_t>> scored;  // (kappa, index into all)
    scored.reserve(all.size());

    for (size_t i = 0; i < all.size(); ++i)
        scored.emplace_back(kappa_with_stats(all[i], t_now, st), i);

    std::sort(scored.begin(), scored.end(),
        [&](const std::pair<double, size_t>& x,
            const std::pair<double, size_t>& y) {
            if (x.first != y.first)
                return x.first > y.first;   // desc kappa
            return key_less(all[x.second].key, all[y.second].key);  // asc
        });

    std::vector<RepairTask> selected;
    std::unordered_map<PartitionKey, RepairTask, PartitionKeyHash> remaining;

    size_t sum = 0;
    bool budget_met = false;

    for (const auto& s : scored) {
        const RepairTask& tsk = all[s.second];

        if (!budget_met) {
            selected.push_back(tsk);
            sum += tsk.occupancy;

            if (sum >= B)
                budget_met = true;   // smallest prefix reaching B
        } else {
            remaining.emplace(tsk.key, tsk);
        }
    }

    _pending.swap(remaining);
    return selected;
}

    // -----------------------------------------------------------------------------
    // per-tenant max-staleness ceiling (SLA hard guarantee)
    // -----------------------------------------------------------------------------

    struct DrainResult {
        std::vector<RepairTask> selected;      // forced ∪ kappa-selected
        size_t forced_count = 0;               // # units force-drained (past ceiling)
        size_t forced_occupancy = 0;           // points in the forced set
        bool forced_overflow = false;          // forced_occupancy > B (contract [D5b-2])
    };

    // [D5b-1] Force-drain past-ceiling units BEFORE kappa-selection: any task
    // with stale(t_now) >= ceiling is force-selected regardless of kappa. The
    // remaining budget B is then filled by descending kappa over the rest (same
    // rule as select_batch). Non-selected tasks are retained in pending.
    //
    // [D5b-2] select-not-resize: the union is ONE combined merge, size <= B,
    // EXCEPT when the forced set alone exceeds B -> drain all forced (size may
    // be > B), still one merge, flag forced_overflow. Never subdivide.
    //
    // Under all-infinite ceilings the forced set is empty and the kappa fill is
    // computed over the full pending set => identical to select_batch (anchor).

    DrainResult select_batch_ceiling(uint64_t t_now, size_t B)
    {
        DrainResult res;

        std::vector<RepairTask> forced;
        std::vector<RepairTask> rest;

        forced.reserve(_pending.size());
        rest.reserve(_pending.size());

        const uint64_t NO_CEIL = std::numeric_limits<uint64_t>::max();

        for (const auto& kv : _pending) {
            const RepairTask& t = kv.second;

            uint64_t st = (t_now >= t.s_w) ? (t_now - t.s_w) : 0;

            if (t.ceiling != NO_CEIL && st >= t.ceiling)
                forced.push_back(t);
            else
                rest.push_back(t);
        }

        size_t sum = 0;

        for (const auto& t : forced) {
            res.selected.push_back(t);
            sum += t.occupancy;
        }

        res.forced_count = forced.size();
        res.forced_occupancy = sum;
        res.forced_overflow = (sum > B);

        std::unordered_map<PartitionKey, RepairTask, PartitionKeyHash> remaining;

        if (sum > B) {
            // Budget already met/exceeded by the forced set -> no kappa fill.
            for (const auto& t : rest)
                remaining.emplace(t.key, t);
        } else {
            // Fill remaining budget by descending kappa over rest.
            TermStats stx = compute_stats_over(rest, t_now);

            std::vector<std::pair<double, size_t>> scored;
            scored.reserve(rest.size());

            for (size_t i = 0; i < rest.size(); ++i)
                scored.emplace_back(kappa_with_stats(rest[i], t_now, stx), i);

            std::sort(
                scored.begin(),
                scored.end(),
                [&](const std::pair<double, size_t>& x,
                    const std::pair<double, size_t>& y) {
                    if (x.first != y.first)
                        return x.first > y.first; // descending kappa

                    return key_less(
                        rest[x.second].key,
                        rest[y.second].key);       // ascending tie-break
                });

            bool budget_met = false;

            for (const auto& s : scored) {
                const RepairTask& tsk = rest[s.second];

                if (!budget_met) {
                    res.selected.push_back(tsk);
                    sum += tsk.occupancy;

                    if (sum >= B)
                        budget_met = true;
                } else {
                    remaining.emplace(tsk.key, tsk);
                }
            }
        }

        _pending.swap(remaining);
        return res;
    }


    size_t pending_count() const {
        return _pending.size();
    }

    // Read-only copy of all pending tasks. Call BEFORE select_batch for drain-time
    // telemetry, since kappa() is normalized over the current pending set.
    std::vector<RepairTask> snapshot() const {
        std::vector<RepairTask> out;
        out.reserve(_pending.size());
        for (const auto& kv : _pending)
            out.push_back(kv.second);
        return out;
    }

    // Composite key for a single task, normalized across the current pending set.
    double kappa(const RepairTask& t, uint64_t t_now) const {
        TermStats st = compute_stats(t_now);
        return kappa_with_stats(t, t_now, st);
    }

    // Stable, canonical text round-trip of config + pending set (tasks sorted by
    // key so serialize() is deterministic). deserialize(serialize()) reproduces
    // the queue exactly. Floats are stored as raw IEEE-754 bit patterns for exact
    // round-trip; doubles use round-trippable precision.	

    std::string serialize() const {
    std::vector<RepairTask> tasks;
    tasks.reserve(_pending.size());

    for (const auto& kv : _pending)
        tasks.push_back(kv.second);

    std::sort(tasks.begin(), tasks.end(),
        [](const RepairTask& a, const RepairTask& b) {
            return key_less(a.key, b.key);
        });

    std::ostringstream os;
    os.precision(std::numeric_limits<double>::max_digits10);

    os << "RSCHED2\n";
    os << _a1 << ' ' << _a2 << ' ' << _a3 << ' ' << _a4 << ' '
       << _B << ' ' << _high_water << ' ' << _low_water << ' '
       << (_dr_enabled ? 1 : 0) << '\n';

    os << tasks.size() << '\n';

    for (const auto& t : tasks) {
        os << t.key.label << ' '
           << t.key.tenant << ' '
           << t.s_w << ' '
           << t.tightest_sla << ' '
           << t.occupancy << ' '
           << float_bits(t.dr_proxy) << ' '
           << t.ceiling << '\n';
    }

    return os.str();
}

void deserialize(const std::string& blob) {
    _pending.clear();

    std::istringstream is(blob);
    std::string magic;

    is >> magic;               // "RSCHED2"
    const bool has_ceiling = (magic == "RSCHED2");
    int dr_flag = 0;
    is >> _a1 >> _a2 >> _a3 >> _a4
       >> _B >> _high_water >> _low_water >> dr_flag;

    _dr_enabled = (dr_flag != 0);

    size_t n;
    is >> n;

    for (size_t i = 0; i < n; ++i) {
        RepairTask t;


        uint32_t bits = 0;

        is >> t.key.label
           >> t.key.tenant
           >> t.s_w
           >> t.tightest_sla
           >> t.occupancy
           >> bits;
        
        if (has_ceiling)
            is >> t.ceiling;

        t.dr_proxy = bits_float(bits);

        _pending.emplace(t.key, t);
    }
}

    private:
        struct TermStats {
            double min_stale = 0, max_stale = 0;
            double min_dr = 0, max_dr = 0;
            double min_sla = 0, max_sla = 0;

            // mig is always 0 -> single distinct value -> always inert.
        };

    static bool key_less(const PartitionKey& a, const PartitionKey& b) {
    if (a.label != b.label)
        return a.label < b.label;
    return a.tenant < b.tenant;
}

    static double stale_raw(const RepairTask& t, uint64_t t_now) {
        return (t_now >= t.s_w) ? static_cast<double>(t_now - t.s_w) : 0.0;
    }

    static double sla_raw(const RepairTask& t) {
        return -static_cast<double>(t.tightest_sla);   // tighter SLO -> larger raw
    }

    static double norm(double v, double lo, double hi) {
        return (hi == lo) ? 0.0 : (v - lo) / (hi - lo);
    }

    TermStats compute_stats_over(const std::vector<RepairTask>& v, uint64_t t_now) const { 
        TermStats st;
        if (v.empty())
            return st;

        double lim = std::numeric_limits<double>::infinity();

        st.min_stale = st.min_dr = st.min_sla = lim;
        st.max_stale = st.max_dr = st.max_sla = -lim;

        for (const auto& t : v) {

            double s = stale_raw(t, t_now);
            double d = static_cast<double>(t.dr_proxy);
            double a = sla_raw(t);

            st.min_stale = std::min(st.min_stale, s);
            st.max_stale = std::max(st.max_stale, s);

            st.min_dr = std::min(st.min_dr, d);
            st.max_dr = std::max(st.max_dr, d);

            st.min_sla = std::min(st.min_sla, a);
            st.max_sla = std::max(st.max_sla, a);
        }

        return st;
        
    }

    TermStats compute_stats(uint64_t t_now) const {
        TermStats st;

        if (_pending.empty())
            return st;

        double lim = std::numeric_limits<double>::infinity();

        st.min_stale = st.min_dr = st.min_sla = lim;
        st.max_stale = st.max_dr = st.max_sla = -lim;

        for (const auto& kv : _pending) {
            const RepairTask& t = kv.second;

            double s = stale_raw(t, t_now);
            double d = static_cast<double>(t.dr_proxy);
            double a = sla_raw(t);

            st.min_stale = std::min(st.min_stale, s);
            st.max_stale = std::max(st.max_stale, s);

            st.min_dr = std::min(st.min_dr, d);
            st.max_dr = std::max(st.max_dr, d);

            st.min_sla = std::min(st.min_sla, a);
            st.max_sla = std::max(st.max_sla, a);
        }

        return st;
    }

    double kappa_with_stats(const RepairTask& t,
                            uint64_t t_now,
                            const TermStats& st) const {
        if (_pending.empty())
            return 0.0;

        double stale_n = norm(stale_raw(t, t_now),
                            st.min_stale, st.max_stale);

        double dr_n = norm(static_cast<double>(t.dr_proxy),
                        st.min_dr, st.max_dr);

        double sla_n = norm(sla_raw(t),
                            st.min_sla, st.max_sla);

        double mig_n = 0.0;   // mig always 0 -> inert

        double dr_term =
            _dr_enabled ? (_a2 * dr_n) : 0.0;

        return _a1 * stale_n +
            dr_term +
            _a3 * sla_n +
            _a4 * mig_n;
    }

    static uint32_t float_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
    }

    static float bits_float(uint32_t b) {
        float f;
        std::memcpy(&f, &b, sizeof(f));
        return f;
    }

    double _a1 = 1.0, _a2 = 0.0, _a3 = 1.0, _a4 = 0.0;   // anchor defaults

    size_t _B = 0, _high_water = 0, _low_water = 0;

    bool _dr_enabled = false;

    std::unordered_map<PartitionKey, RepairTask, PartitionKeyHash> _pending;

};

} // namespace diskann