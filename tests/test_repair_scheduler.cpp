// M5.3 key-order unit test for RepairScheduler (Agent 1, C3).
// Standalone: compile directly, links nothing (header-only + partition_key.h).
//
//   g++ -std=c++17 -O2 -Iinclude tests/test_repair_scheduler.cpp \
//       -o /tmp/test_repair_scheduler && /tmp/test_repair_scheduler

#include <cassert>
#include <cstdio>
#include <vector>
#include <iostream>
#include <limits>
#include <cstring>

#include "v2/repair_scheduler.h"

using namespace diskann;

static bool s_kappa_geq(const std::vector<RepairTask>& sel, size_t i, size_t j,
                        uint64_t t_now);

static RepairTask mk(uint32_t label, uint32_t tenant, uint64_t s_w,
                     uint32_t sla, size_t occ, float dr = 0.0f) {
  RepairTask t;
  t.key.label = label;
  t.key.tenant = tenant;
  t.s_w = s_w;
  t.tightest_sla = sla;
  t.occupancy = occ;
  t.dr_proxy = dr;
  return t;
}

// Test 1: descending kappa + deterministic tie-break, budget large => all.
static void test_order_and_tiebreak() {
  RepairScheduler s;

  s.configure(/*a1=*/1, /*a2=*/0, /*a3=*/1, /*a4=*/0,
              /*B=*/0, /*high_water=*/0, /*low_water=*/0,
              /*dr_enabled=*/false);

  // uniform SLA's => sla term inert; order driven by staleness.
  s.push_or_update(mk(1, 0, /*s_w=*/100, 5, 10));  // stale 200
  s.push_or_update(mk(2, 0, /*s_w=*/50, 5, 10));   // stale 250 (tie w/ label3)
  s.push_or_update(mk(3, 0, /*s_w=*/50, 5, 10));   // stale 250
  s.push_or_update(mk(0, 0, /*s_w=*/200, 5, 10));  // stale 100 (newest)

  const uint64_t t_now = 300;
  auto sel = s.select_batch(t_now, /*B=*/1000);  // budget > total => all 4

  assert(sel.size() == 4);

  // expected: label2, label3 (tie, key asc), label1, label0
  assert(sel[0].key.label == 2);
  assert(sel[1].key.label == 3);
  assert(sel[2].key.label == 1);
  assert(sel[3].key.label == 0);

  // kappa non-increasing along the returned order.
  for (size_t i = 1; i < sel.size(); ++i)
    assert(s_kappa_geq(sel, i - 1, i, t_now));

  assert(s.pending_count() == 0);  // all drained

  std::printf("[PASS] test_order_and_tiebreak\n");
}

// helper: recompute kappa on a scratch scheduler holding the same set.
static bool s_kappa_geq(const std::vector<RepairTask>& sel, size_t i, size_t j,
                        uint64_t t_now) {
  RepairScheduler tmp;

  tmp.configure(1, 0, 1, 0, 0, 0, 0, false);

  for (const auto& t : sel)
    tmp.push_or_update(t);

  return tmp.kappa(sel[i], t_now) >= tmp.kappa(sel[j], t_now);
}

// Test 2: budget = smallest prefix with summed occupancy >= B; retain rest.
static void test_budget() {
  RepairScheduler s;

  s.configure(1, 0, 1, 0, 0, 0, 0, false);

  s.push_or_update(mk(1, 0, 100, 5, 10));  // stale 200
  s.push_or_update(mk(2, 0, 50, 5, 10));   // stale 250
  s.push_or_update(mk(3, 0, 50, 5, 10));   // stale 250
  s.push_or_update(mk(0, 0, 200, 5, 10));  // stale 100

  auto sel = s.select_batch(/*t_now=*/300, /*B=*/25);

  // order [2,3,1,0]; prefix sums 10,20,30 -> stop at 3 tasks (sum 30 >= 25).
  assert(sel.size() == 3);

  size_t sum = 0;
  for (const auto& t : sel)
    sum += t.occupancy;

  assert(sum >= 25);
  assert(sum - sel.back().occupancy < 25);  // truly the smallest prefix

  assert(s.pending_count() == 1);  // label0 retained

  std::printf("[PASS] test_budget\n");
}

// Test 3: serialize/deserialize round-trip (same tasks, same s_w).
static void test_roundtrip()  // --- Test 3: round-trip asserts per-task VALUES
                              // + keys (not just count) ---
{
  RepairScheduler a;
  a.configure(1.0, 0.0, 1.0, 0.0, 500, 1000, 0, false);
  auto mkc = [](uint32_t lab, uint32_t ten, uint64_t s_w, uint32_t sla,
                size_t occ, float dr, uint64_t ceil) {
    RepairTask t;
    t.key = {lab, ten};
    t.s_w = s_w;
    t.tightest_sla = sla;
    t.occupancy = occ;
    t.dr_proxy = dr;
    t.ceiling = ceil;
    return t;
  };
  a.push_or_update(mkc(1, 2, 100, 7, 50, 1.5f, 10000));
  a.push_or_update(mkc(3, 4, 200, 9, 111, 2.5f, 30000));
  a.push_or_update(
      mkc(5, 6, 300, 3, 22, 0.5f, std::numeric_limits<uint64_t>::max()));

  RepairScheduler b;
  b.deserialize(a.serialize());
  assert(a.pending_count() == b.pending_count());
  auto sa = a.snapshot(), sb = b.snapshot();
  auto bk = [](const RepairTask& x, const RepairTask& y) {
    if (x.key.label != y.key.label)
      return x.key.label < y.key.label;
    return x.key.tenant < y.key.tenant;
  };
  std::sort(sa.begin(), sa.end(), bk);
  std::sort(sb.begin(), sb.end(), bk);
  for (size_t i = 0; i < sa.size(); ++i) {
    assert(sa[i].key.label == sb[i].key.label);
    assert(sa[i].key.tenant == sb[i].key.tenant);
    assert(sa[i].s_w == sb[i].s_w);
    assert(sa[i].tightest_sla == sb[i].tightest_sla);
    assert(sa[i].occupancy == sb[i].occupancy);
    assert(sa[i].ceiling == sb[i].ceiling);
    assert(std::memcmp(&sa[i].dr_proxy, &sb[i].dr_proxy, sizeof(float)) == 0);
  }
  std::cout << "[test] value+key round-trip OK\n";
}
// Test 4: defaults + uniform SLA => ordering purely by staleness (FIFO).
static void test_fifo_collapse() {
  RepairScheduler s;
  s.configure(1, 0, 1, 0, 0, 0, 0, false);

  // push in arbitrary order; distinct s_w; uniform sla.
  s.push_or_update(mk(9, 0, 300, 7, 5));
  s.push_or_update(mk(9, 1, 100, 7, 5));  // oldest
  s.push_or_update(mk(9, 2, 200, 7, 5));

  auto sel = s.select_batch(/*t_now=*/1000, /*B=*/1000);
  assert(sel.size() == 3);

  // older s_w first: 100, 200, 300
  assert(sel[0].s_w == 100);
  assert(sel[1].s_w == 200);
  assert(sel[2].s_w == 300);

  std::printf("[PASS] test_fifo_collapse\n");
}

// Test 5: push_or_update keeps EARLIEST s_w, refreshes other fields.
static void test_upsert_keeps_earliest_s_w() {
  RepairScheduler s;
  s.configure(1, 0, 1, 0, 0, 0, 0, false);

  s.push_or_update(mk(5, 5, /*s_w=*/500, 3, 10, 10.0f));
  s.push_or_update(mk(5, 5, /*s_w=*/900, 2, 33, 33.0f));  // touched again

  assert(s.pending_count() == 1);

  auto sel = s.select_batch(/*t_now=*/1000, /*B=*/1000);
  assert(sel.size() == 1);

  assert(sel[0].s_w == 500);         // earliest kept
  assert(sel[0].occupancy == 33);    // latest refreshed
  assert(sel[0].tightest_sla == 2);  // latest refreshed

  std::printf("[PASS] test_upsert_keeps_earliest_s_w\n");
}

int main() {
  test_order_and_tiebreak();
  test_budget();
  test_roundtrip();
  test_fifo_collapse();
  test_upsert_keeps_earliest_s_w();

  std::printf("ALL TESTS PASSED\n");
  return 0;
}