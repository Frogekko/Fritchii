// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numeric>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/host_memory.h"
#include "common/logging.h"
#include "common/nx2_hitch.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace Common::Nx2Hitch {

namespace {

constexpr long long ThresholdUs = 50'000;
constexpr size_t Causes = static_cast<size_t>(Cause::Count);
constexpr size_t MaxThreads = 48;

struct ThreadEntry {
    char name[32]{};
#ifdef _WIN32
    HANDLE handle{};
#endif
};

std::array<ThreadEntry, MaxThreads> g_threads;
std::atomic<size_t> g_thread_count{};
std::mutex g_thread_mutex;

struct Snapshot {
    std::array<u64, Causes> ns{};
    std::array<u64, Causes> count{};
    std::array<u64, Causes> amount{};
    std::array<u64, Causes> longest{};
    u64 page_faults{};
    u64 zero_filled{};
    u64 user_100ns{};   ///< the process's CPU time in user mode, all threads
    u64 kernel_100ns{}; ///< and in kernel mode (page faults, VM calls)
    size_t threads{};
    std::array<u64, MaxThreads> thread_user{};
    std::array<u64, MaxThreads> thread_kernel{};
};

#ifdef _WIN32
u64 FileTimeValue(const FILETIME& time) {
    return (u64{time.dwHighDateTime} << 32) | time.dwLowDateTime;
}
#endif

Snapshot Take() {
    Snapshot snapshot;
    for (size_t i = 0; i < Causes; ++i) {
        snapshot.ns[i] = g_counters.ns[i].load(std::memory_order_relaxed);
        snapshot.count[i] = g_counters.count[i].load(std::memory_order_relaxed);
        snapshot.amount[i] = g_counters.amount[i].load(std::memory_order_relaxed);
        snapshot.longest[i] = g_counters.longest[i].exchange(0, std::memory_order_relaxed);
    }
    snapshot.zero_filled = HostMemory::ZeroFilledBytes();
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        snapshot.page_faults = counters.PageFaultCount;
    }
    FILETIME created{}, exited{}, kernel{}, user{};
    if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
        snapshot.user_100ns = FileTimeValue(user);
        snapshot.kernel_100ns = FileTimeValue(kernel);
    }
    snapshot.threads = g_thread_count.load(std::memory_order_acquire);
    for (size_t i = 0; i < snapshot.threads; ++i) {
        if (GetThreadTimes(g_threads[i].handle, &created, &exited, &kernel, &user)) {
            snapshot.thread_user[i] = FileTimeValue(user);
            snapshot.thread_kernel[i] = FileTimeValue(kernel);
        }
    }
#endif
    return snapshot;
}

bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_HITCH_LOG");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// The busiest named threads in the gap, CPU time summed per name (worker pools share one).
std::string BusiestThreads(const Snapshot& last, const Snapshot& now) {
    struct Busy {
        std::string name;
        u64 user{};
        u64 kernel{};
    };
    std::vector<Busy> busy;
    for (size_t i = 0; i < now.threads; ++i) {
        const u64 user = now.thread_user[i] - (i < last.threads ? last.thread_user[i] : 0);
        const u64 kernel = now.thread_kernel[i] - (i < last.threads ? last.thread_kernel[i] : 0);
        if (user + kernel == 0) {
            continue;
        }
        const std::string name{g_threads[i].name};
        const auto it = std::ranges::find(busy, name, &Busy::name);
        if (it != busy.end()) {
            it->user += user;
            it->kernel += kernel;
        } else {
            busy.push_back({name, user, kernel});
        }
    }
    std::ranges::sort(busy, [](const Busy& a, const Busy& b) {
        return a.user + a.kernel > b.user + b.kernel;
    });
    std::string text;
    for (size_t i = 0; i < (std::min)(busy.size(), size_t{6}); ++i) {
        text += fmt::format("{}{} {:.0f}+{:.0f}", i == 0 ? "" : ", ", busy[i].name,
                            static_cast<double>(busy[i].user) / 1e4,
                            static_cast<double>(busy[i].kernel) / 1e4);
    }
    return text.empty() ? "none measured" : text;
}

} // Anonymous namespace

namespace {
std::mutex g_compile_mutex;
std::unordered_map<u64, u8> g_compiled_on; ///< key -> mask of the cores that compiled it
u8 g_cores_seen{};
// Taken from `head`: erasing 16 keys from the front of an 800k-key vector moved ~6 MiB under
// g_compile_mutex, which every on-demand compile on every core waits for (2026-11-16).
struct WarmLane {
    std::vector<u64> keys;
    size_t head{};
    size_t Size() const {
        return keys.size() - head;
    }
};
// `fresh` (what another core compiled this session, so the game runs it now) goes before `saved`
// (the persisted set): behind 800k saved keys it waited minutes (review, 2026-11-16).
struct WarmQueue {
    WarmLane fresh;
    WarmLane saved;
    size_t Size() const {
        return fresh.Size() + saved.Size();
    }
};
std::array<WarmQueue, 8> g_warm_queues;
std::atomic<bool> g_warm_boost{}; ///< a saved set is queued and nothing was presented yet
constexpr size_t WarmQueueCap = 1'200'000; ///< a saved set (up to PersistCap) plus new code
u8 g_warm_full{}; ///< cores whose JIT refused a warm compile for lack of code space

// EDEN_NX2_JIT_WARM_PERSIST. A key is an A64 LocationDescriptor: PC in bits 0-55, FPCR mode bits
// and the single-step bit above. Saved keys hold the PC as an offset from the code base.
constexpr u64 PcMask = (u64{1} << 56) - 1;
constexpr size_t PersistCap = 1'000'000; ///< 16 MiB; past it the least recently compiled go
constexpr auto SaveInterval = std::chrono::minutes{5};
constexpr u32 PersistVersion = 1;
constexpr std::array<char, 8> PersistMagic{'F', 'R', 'J', 'I', 'T', 'W', 'P', '1'};

struct PersistEntry {
    u64 rel;       ///< key with the PC as an offset from the code base
    u32 last_seen; ///< the last session a core compiled it on demand
    u16 uses;      ///< sessions a core compiled it on demand
    u8 mask;       ///< cores that compiled it on demand
    u8 reserved;
};
static_assert(sizeof(PersistEntry) == 16);

struct PersistHeader {
    std::array<char, 8> magic;
    u32 version;
    u32 entry_size;
    u64 layout_fingerprint;
    u64 code_size;
    u64 checksum; ///< of the entries
    u32 session;
    u32 count;
};
static_assert(sizeof(PersistHeader) == 48);

struct Persist {
    bool active{};
    std::filesystem::path file;
    u64 base{};
    u64 size{};
    u64 fingerprint{};
    u32 session{};
    std::vector<PersistEntry> entries;     ///< first-compile order
    std::unordered_map<u64, u32> index{}; ///< key -> entries index
};
Persist g_persist; ///< under g_compile_mutex
std::mutex g_save_mutex;
std::mutex g_saver_mutex;
std::condition_variable_any g_saver_cv;
std::jthread g_saver; ///< after what it waits on, so an exit without EndSession joins it first

u64 Checksum(const std::vector<PersistEntry>& entries) {
    u64 hash = 0xcbf29ce484222325ULL;
    for (const PersistEntry& entry : entries) {
        u64 words[2];
        std::memcpy(words, &entry, sizeof(words));
        hash = (hash ^ words[0]) * 0x100000001b3ULL;
        hash = (hash ^ words[1]) * 0x100000001b3ULL;
    }
    return hash;
}

// Called with g_compile_mutex held, for on-demand compiles only: a warm compile says nothing
// about whether the game still runs the block.
void NotePersistLocked(u64 key, u8 bit) {
    if (((key >> 57) & 1) != 0) {
        return;
    }
    const u64 pc = key & PcMask;
    if (pc - g_persist.base >= g_persist.size) {
        return;
    }
    const auto [it, inserted] =
        g_persist.index.try_emplace(key, static_cast<u32>(g_persist.entries.size()));
    if (inserted) {
        g_persist.entries.push_back(
            {(pc - g_persist.base) | (key & ~PcMask), g_persist.session, 1, bit, 0});
        return;
    }
    PersistEntry& entry = g_persist.entries[it->second];
    entry.mask |= bit;
    if (entry.last_seen != g_persist.session) {
        entry.last_seen = g_persist.session;
        if (entry.uses != 0xFFFF) {
            ++entry.uses;
        }
    }
}

std::vector<PersistEntry> LoadPersistFile(const std::filesystem::path& file, u64 fingerprint,
                                          u64 code_size, u32& session, std::string& problem) {
    std::ifstream in{file, std::ios::binary};
    if (!in) {
        problem = "no file yet";
        return {};
    }
    PersistHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || header.magic != PersistMagic || header.version != PersistVersion ||
        header.entry_size != sizeof(PersistEntry)) {
        problem = "not a cache file of this version";
        return {};
    }
    if (header.layout_fingerprint != fingerprint || header.code_size != code_size) {
        problem = fmt::format("the module layout changed (fingerprint {:016X} -> {:016X}, code "
                              "size {:#x} -> {:#x})",
                              header.layout_fingerprint, fingerprint, header.code_size, code_size);
        return {};
    }
    if (header.count > PersistCap) {
        // The header has no checksum: a damaged count would ask for up to 64 GiB.
        problem = "truncated or corrupt";
        return {};
    }
    std::vector<PersistEntry> entries(header.count);
    in.read(reinterpret_cast<char*>(entries.data()),
            static_cast<std::streamsize>(entries.size() * sizeof(PersistEntry)));
    if (!in || in.peek() != std::char_traits<char>::eof() ||
        Checksum(entries) != header.checksum) {
        problem = "truncated or corrupt";
        return {};
    }
    session = header.session;
    return entries;
}

void SavePersist() {
    const std::scoped_lock save_lock{g_save_mutex};
    std::vector<PersistEntry> entries;
    std::filesystem::path file;
    PersistHeader header{};
    header.magic = PersistMagic;
    header.version = PersistVersion;
    header.entry_size = static_cast<u32>(sizeof(PersistEntry));
    {
        const std::scoped_lock lock{g_compile_mutex};
        if (!g_persist.active) {
            return;
        }
        entries = g_persist.entries;
        file = g_persist.file;
        header.layout_fingerprint = g_persist.fingerprint;
        header.code_size = g_persist.size;
        header.session = g_persist.session;
    }
    size_t dropped = 0;
    if (entries.size() > PersistCap) {
        // Keep the most recently compiled, then the most used; the kept ones stay in order.
        std::vector<u32> rank(entries.size());
        std::iota(rank.begin(), rank.end(), 0U);
        std::ranges::nth_element(rank, rank.begin() + static_cast<std::ptrdiff_t>(PersistCap),
                                 [&](u32 a, u32 b) {
            const PersistEntry& x = entries[a];
            const PersistEntry& y = entries[b];
            if (x.last_seen != y.last_seen) {
                return x.last_seen > y.last_seen;
            }
            return x.uses != y.uses ? x.uses > y.uses : a < b;
        });
        std::vector<bool> keep(entries.size());
        for (size_t i = 0; i < PersistCap; ++i) {
            keep[rank[i]] = true;
        }
        std::vector<PersistEntry> kept;
        kept.reserve(PersistCap);
        for (size_t i = 0; i < entries.size(); ++i) {
            if (keep[i]) {
                kept.push_back(entries[i]);
            }
        }
        dropped = entries.size() - kept.size();
        entries = std::move(kept);
    }
    header.count = static_cast<u32>(entries.size());
    header.checksum = Checksum(entries);

    // Written beside the file and renamed over it: a crash leaves the previous file whole.
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    std::filesystem::path temp = file;
    temp += ".tmp";
    {
        std::ofstream out{temp, std::ios::binary | std::ios::trunc};
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.write(reinterpret_cast<const char*>(entries.data()),
                  static_cast<std::streamsize>(entries.size() * sizeof(PersistEntry)));
        out.flush();
        if (!out) {
            LOG_WARNING(Core_ARM, "JIT warm persist: cannot write {}", temp.string());
            return;
        }
    }
    std::filesystem::rename(temp, file, ec);
    if (ec) {
        LOG_WARNING(Core_ARM, "JIT warm persist: cannot replace {}: {}", file.string(),
                    ec.message());
        return;
    }
    LOG_INFO(Core_ARM, "JIT warm persist: saved {} blocks (session {}){}", entries.size(),
             header.session,
             dropped != 0 ? fmt::format(", dropped the {} least recently compiled", dropped)
                          : "");
}

// Session totals: the three compile causes plus the presents' long gaps, from BeginSession.
struct CompileTotals {
    std::array<u64, 3> count{};
    std::array<u64, 3> ns{};
};
constexpr std::array CompileCauses{Cause::Recompile, Cause::RecompileRepeat, Cause::WarmCompile};

CompileTotals ReadCompileTotals() {
    CompileTotals totals;
    for (size_t i = 0; i < CompileCauses.size(); ++i) {
        const auto cause = static_cast<size_t>(CompileCauses[i]);
        totals.count[i] = g_counters.count[cause].load(std::memory_order_relaxed);
        totals.ns[i] = g_counters.ns[cause].load(std::memory_order_relaxed);
    }
    return totals;
}

std::mutex g_session_mutex;
CompileTotals g_session_start; ///< under g_session_mutex
constexpr u64 SummaryIntervalNs =
    static_cast<u64>(std::chrono::nanoseconds{SaveInterval}.count());
std::atomic<u64> g_session_begin_ns{};
std::atomic<u64> g_next_summary_ns{};
std::atomic<bool> g_first_present_pending{};
std::atomic<u64> g_hitch_count{};
std::atomic<u64> g_hitch_us{};
std::atomic<u64> g_hitch_longest_us{};

void LogSessionTotals(std::string_view when) {
    const u64 begin = g_session_begin_ns.load(std::memory_order_acquire);
    if (begin == 0) {
        return;
    }
    CompileTotals start;
    {
        const std::scoped_lock lock{g_session_mutex};
        start = g_session_start;
    }
    const CompileTotals now = ReadCompileTotals();
    size_t waiting = 0;
    u8 full = 0;
    {
        const std::scoped_lock lock{g_compile_mutex};
        for (const auto& queue : g_warm_queues) {
            waiting += queue.Size();
        }
        full = g_warm_full;
    }
    const auto ms = [&](size_t i) { return static_cast<double>(now.ns[i] - start.ns[i]) / 1e6; };
    const auto count = [&](size_t i) { return now.count[i] - start.count[i]; };
    LOG_INFO(Core_ARM,
             "JIT session totals {} ({:.1f} min): new game code {} compiles ({:.0f} ms), code "
             "another core had {} ({:.0f} ms), warm compiles {} ({:.0f} ms), {} keys waiting to "
             "be warmed, cores out of warm code space {:#04x}; gaps over 50 ms between presents "
             "{} ({:.0f} ms, longest {:.0f} ms)",
             when, static_cast<double>(NowNs() - begin) / 60e9, count(0), ms(0), count(1), ms(1),
             count(2), ms(2), waiting, full, g_hitch_count.load(std::memory_order_relaxed),
             static_cast<double>(g_hitch_us.load(std::memory_order_relaxed)) / 1e3,
             static_cast<double>(g_hitch_longest_us.load(std::memory_order_relaxed)) / 1e3);
}
} // Anonymous namespace

bool JitWarmEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_JIT_WARM");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

bool JitWarmPersistEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_JIT_WARM_PERSIST");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

void BeginSession() {
    {
        const std::scoped_lock lock{g_session_mutex};
        g_session_start = ReadCompileTotals();
    }
    g_hitch_count.store(0, std::memory_order_relaxed);
    g_hitch_us.store(0, std::memory_order_relaxed);
    g_hitch_longest_us.store(0, std::memory_order_relaxed);
    const u64 now = NowNs();
    g_next_summary_ns.store(now + SummaryIntervalNs, std::memory_order_relaxed);
    g_first_present_pending.store(true, std::memory_order_relaxed);
    g_session_begin_ns.store(now, std::memory_order_release);
}

void BeginWarmPersist(const std::filesystem::path& file, u64 code_base, u64 code_size,
                      u64 layout_fingerprint) {
    u32 last_session = 0;
    std::string problem;
    std::vector<PersistEntry> loaded =
        LoadPersistFile(file, layout_fingerprint, code_size, last_session, problem);
    size_t queued_own = 0;
    size_t queued = 0;
    size_t loaded_count = 0;
    u8 cores = 0;
    {
        const std::scoped_lock lock{g_compile_mutex};
        // A new process: its JITs start empty, and the last game's keys mean nothing here.
        g_compiled_on.clear();
        // Sized now: growing to ~1M keys rehashes several times under every compile's lock.
        g_compiled_on.reserve(loaded.size() + loaded.size() / 4);
        g_cores_seen = 0;
        g_warm_full = 0;
        for (auto& queue : g_warm_queues) {
            queue = {};
        }
        g_persist = {};
        g_persist.file = file;
        g_persist.base = code_base;
        g_persist.size = code_size;
        g_persist.fingerprint = layout_fingerprint;
        g_persist.session = last_session + 1;
        g_persist.entries = std::move(loaded);
        g_persist.entries.reserve(g_persist.entries.size() + 65536); // new blocks: no 13 MiB copy
        g_persist.index.reserve(g_persist.entries.size());
        for (const PersistEntry& entry : g_persist.entries) {
            cores |= entry.mask;
        }
        // Every block goes to every core the game ran code on, not only the ones that compiled
        // it: per-core queues left ~1.1M compiles of code another core had (2026-11-15), as the
        // guest's threads land on different cores each session. First the cores that compiled
        // it, then the others: with every core warming the same keys in the same order, the
        // save-load code was still unwarmed when the game ran it (new-code compiles 364k ->
        // 651k in the first 5 minutes, 14 -> 57 s of gaps, 2026-11-16). First-compile order
        // within each pass, so the boot and save-load code comes first.
        for (const bool own : {true, false}) {
            for (u32 i = 0; i < g_persist.entries.size(); ++i) {
                const PersistEntry& entry = g_persist.entries[i];
                const u64 offset = entry.rel & PcMask;
                if (offset >= code_size) {
                    continue;
                }
                const u64 key = (code_base + offset) | (entry.rel & ~PcMask);
                if (own) {
                    g_persist.index.emplace(key, i);
                }
                const u8 targets = own ? entry.mask : static_cast<u8>(cores & ~entry.mask);
                for (u32 core = 0; core < 8; ++core) {
                    if (((targets >> core) & 1) != 0 && g_warm_queues[core].Size() < WarmQueueCap) {
                        g_warm_queues[core].saved.keys.push_back(key);
                        ++(own ? queued_own : queued);
                    }
                }
            }
        }
        loaded_count = g_persist.entries.size();
        g_persist.active = true;
    }
    g_warm_boost.store(queued_own + queued != 0, std::memory_order_relaxed);
    if (problem.empty()) {
        LOG_INFO(Core_ARM,
                 "JIT warm persist: {} blocks from {} (session {}), {} queued for the cores that "
                 "compiled them, then {} for the other cores of {:#04x}; game code at {:#x}, "
                 "{} KiB",
                 loaded_count, file.string(), last_session + 1, queued_own, queued, cores,
                 code_base, code_size >> 10);
    } else {
        LOG_INFO(Core_ARM, "JIT warm persist: starting empty, {}: {} (game code at {:#x}, {} KiB)",
                 problem, file.string(), code_base, code_size >> 10);
    }
    g_saver = std::jthread([](std::stop_token stop) {
        std::unique_lock lock{g_saver_mutex};
        while (!g_saver_cv.wait_for(lock, stop, SaveInterval, [] { return false; }) &&
               !stop.stop_requested()) {
            lock.unlock();
            SavePersist();
            lock.lock();
        }
    });
}

void EndSession() {
    g_warm_boost.store(false, std::memory_order_relaxed);
    if (g_saver.joinable()) {
        g_saver.request_stop();
        g_saver.join();
    }
    SavePersist();
    {
        const std::scoped_lock lock{g_compile_mutex};
        g_persist = {};
    }
    LogSessionTotals("at the end");
    g_session_begin_ns.store(0, std::memory_order_release);
}

Cause CompileCause(u64 key, u32 core) {
    const u8 bit = static_cast<u8>(1U << (core & 7));
    std::scoped_lock lock{g_compile_mutex};
    g_cores_seen |= bit;
    u8& mask = g_compiled_on[key];
    const bool repeat = (mask & ~bit) != 0;
    mask |= bit;
    if (t_warming) {
        return Cause::WarmCompile;
    }
    if (g_persist.active) {
        NotePersistLocked(key, bit);
    }
    // Single-stepping blocks (bit 57 of an A64 LocationDescriptor) are never worth warming.
    if (JitWarmEnabled() && ((key >> 57) & 1) == 0) {
        for (u32 other = 0; other < 8; ++other) {
            const u8 other_bit = static_cast<u8>(1U << other);
            if ((g_cores_seen & other_bit) != 0 && (mask & other_bit) == 0 &&
                (g_warm_full & other_bit) == 0 && g_warm_queues[other].Size() < WarmQueueCap) {
                g_warm_queues[other].fresh.keys.push_back(key);
            }
        }
    }
    return repeat ? Cause::RecompileRepeat : Cause::Recompile;
}

size_t TakeWarmKeys(u32 core, std::vector<u64>& out, size_t max) {
    out.clear();
    const u8 bit = static_cast<u8>(1U << (core & 7));
    // Oldest first; skip keys this core compiled on demand meanwhile. A core that ran the boot
    // or a load on demand has runs of 100k+ such keys in a saved set: the lock is dropped every
    // 1024 keys and the call gives up after 16384, so no compile waits long behind the scan.
    for (size_t scanned = 0; scanned < 16384;) {
        std::scoped_lock lock{g_compile_mutex};
        auto& queue = g_warm_queues[core & 7];
        WarmLane& lane = queue.fresh.Size() != 0 ? queue.fresh : queue.saved;
        const size_t stop = scanned + 1024;
        while (lane.head < lane.keys.size() && out.size() < max && scanned < stop) {
            const u64 key = lane.keys[lane.head++];
            ++scanned;
            if ((g_compiled_on[key] & bit) == 0) {
                out.push_back(key);
            }
        }
        if (lane.head == lane.keys.size()) {
            lane = {};
        } else if (lane.head >= 65536 && lane.head * 2 >= lane.keys.size()) {
            lane.keys.erase(lane.keys.begin(),
                            lane.keys.begin() + static_cast<std::ptrdiff_t>(lane.head));
            lane.head = 0;
        }
        if (queue.Size() == 0 || out.size() >= max) {
            break;
        }
    }
    return out.size();
}

void WarmOutOfSpace(u32 core) {
    const u8 bit = static_cast<u8>(1U << (core & 7));
    size_t dropped = 0;
    {
        std::scoped_lock lock{g_compile_mutex};
        if ((g_warm_full & bit) != 0) {
            return;
        }
        g_warm_full |= bit;
        auto& queue = g_warm_queues[core & 7];
        dropped = queue.Size();
        queue = {};
    }
    LOG_INFO(Core_ARM,
             "JIT warm: core {} has under 256 MiB of code space left, so it stops warming; "
             "dropped {} queued keys",
             core, dropped);
}

void RegisterThread(const char* name) {
    const std::string_view view{name};
    t_role = view.starts_with("CPUCore") ? Role::GuestCore
             : view == "GPUBack"         ? Role::GpuBack
             : view == "GPU"             ? Role::GpuFront
                                         : Role::Other;
#ifdef _WIN32
    std::scoped_lock lock{g_thread_mutex};
    const size_t index = g_thread_count.load(std::memory_order_relaxed);
    if (index >= MaxThreads) {
        return;
    }
    HANDLE handle{};
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
                         THREAD_QUERY_LIMITED_INFORMATION, FALSE, 0)) {
        return;
    }
    ThreadEntry& entry = g_threads[index];
    std::strncpy(entry.name, name, sizeof(entry.name) - 1);
    entry.handle = handle;
    g_thread_count.store(index + 1, std::memory_order_release);
#else
    static_cast<void>(name);
#endif
}

bool WarmBoost() {
    return g_warm_boost.load(std::memory_order_relaxed);
}

void OnPresent(long long gap_us) {
    static Snapshot last = Take();
    static u64 window_start_ns = NowNs();
    static int lines_in_window = 0;
    if (g_warm_boost.load(std::memory_order_relaxed)) {
        g_warm_boost.store(false, std::memory_order_relaxed);
    }
    if (!Enabled()) {
        return;
    }
    if (const u64 begin = g_session_begin_ns.load(std::memory_order_acquire); begin != 0) {
        const u64 now_ns = NowNs();
        if (g_first_present_pending.exchange(false, std::memory_order_relaxed)) {
            LOG_INFO(Core_ARM, "JIT session: first present {:.2f} s after boot",
                     static_cast<double>(now_ns - begin) / 1e9);
            LogSessionTotals("at the first present"); // how much warmed before a frame showed
        }
        if (gap_us >= ThresholdUs) {
            const auto gap = static_cast<u64>(gap_us);
            g_hitch_count.fetch_add(1, std::memory_order_relaxed);
            g_hitch_us.fetch_add(gap, std::memory_order_relaxed);
            if (gap > g_hitch_longest_us.load(std::memory_order_relaxed)) {
                g_hitch_longest_us.store(gap, std::memory_order_relaxed);
            }
        }
        if (now_ns >= g_next_summary_ns.load(std::memory_order_relaxed)) {
            g_next_summary_ns.store(now_ns + SummaryIntervalNs, std::memory_order_relaxed);
            LogSessionTotals("so far");
        }
    }
    const Snapshot now = Take();
    if (gap_us >= ThresholdUs) {
        // At most 60 lines a minute (a loading screen can stall many presents in a row; 20 hid a
        // 740 ms first load behind the title screen's gaps, 2026-11-10).
        const u64 now_ns = NowNs();
        if (now_ns - window_start_ns >= 60'000'000'000ULL) {
            window_start_ns = now_ns;
            lines_in_window = 0;
        }
        if (lines_in_window++ < 60) {
            std::string causes;
            for (size_t i = 0; i < Causes; ++i) {
                const u64 ns = now.ns[i] - last.ns[i];
                const u64 count = now.count[i] - last.count[i];
                if (count == 0) {
                    continue;
                }
                const u64 amount = now.amount[i] - last.amount[i];
                causes += fmt::format("{}{} {:.1f} ms ({}{}, longest {:.1f})",
                                      causes.empty() ? "" : ", ", CauseNames[i],
                                      static_cast<double>(ns) / 1e6, count,
                                      amount != 0 ? fmt::format(", {} MiB", amount >> 20) : "",
                                      static_cast<double>(now.longest[i]) / 1e6);
            }
            const u64 trim_since = g_counters.zero_trim_since.load(std::memory_order_relaxed);
            if (trim_since != 0) {
                causes += fmt::format("{}a zero-page pass running for {:.0f} ms",
                                      causes.empty() ? "" : ", ",
                                      static_cast<double>(now_ns - trim_since) / 1e6);
            }
            LOG_INFO(Render, // with the frame lines it lines up against
                     "Hitch: {:.1f} ms between presents; in it: {}; page faults {}, kernel "
                     "zero-filled {} MiB, CPU {:.0f} ms user + {:.0f} ms kernel; busiest threads "
                     "(user+kernel ms): {}",
                     static_cast<double>(gap_us) / 1e3, causes.empty() ? "nothing counted" : causes,
                     now.page_faults - last.page_faults, (now.zero_filled - last.zero_filled) >> 20,
                     static_cast<double>(now.user_100ns - last.user_100ns) / 1e4,
                     static_cast<double>(now.kernel_100ns - last.kernel_100ns) / 1e4,
                     BusiestThreads(last, now));
        }
    }
    last = now;
}

} // namespace Common::Nx2Hitch
